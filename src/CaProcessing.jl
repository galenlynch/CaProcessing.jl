module CaProcessing

import Base.Threads.@spawn

import Base: getindex, firstindex, lastindex, size, setindex!, IndexStyle,
    axes, axes1, IdentityUnitRange

# Stdlib
using Mmap, Statistics
# Public packages
using FixedPointNumbers: Normed, N0f8
using ImageCore: rawview
# Private packages
using GLUtilities: indices_above_thresh

export pixel_lut,
    avg_intensities,
    clip_imgs,
    clip_segments_thr,
    demin,
    demin!,
    find_segments_thr,
    map_to_8bit,
    max_intensities,
    srgb_gamma_compress,
    srgb_gamma_expand,
    rescale_compress,
    rescale_compress_img,
    rescale_compress_img!

struct PixelLUT{T} <: AbstractArray{T,1}
    vals::Vector{T}
    lowerbnd::Int
end
PixelLUT(vals, lowerbnd::Integer) = PixelLUT(vals, convert(Int, lowerbnd))

function pixel_lut(f, ::Type{T}, r::AbstractUnitRange, firstval, lastval) where T
    vals = Vector{T}(undef, length(r) + 2)
    @inbounds vals[1] = firstval
    @inbounds for (i, x) in enumerate(r)
        vals[i + 1] = f(x)
    end
    @inbounds vals[end] = lastval
    PixelLUT(vals, first(r))
end

function pixel_lut(f, r::AbstractUnitRange)
    isempty(r) && throw(ArgumentError("Must specify type and end values if r is empty"))
    firstval = f(first(r))
    lastval = f(last(r))
    T = typeof(firstval)
    pixel_lut(f, T, r, firstval, lastval)
end

pixel_lut(f, i::T) where T<:Integer = pixel_lut(f, zero(T):i)

function pixel_lut(raw_vals::AbstractVector{T}, lowerbnd = 0) where T
    nraw = length(raw_vals)
    nraw > 0 || throw(ArgumentError("raw_vals cannot be empty"))
    vals = similar(raw_vals, (nraw + 2,))
    @inbounds vals[1] = raw_vals[1]
    unsafe_copyto!(vals, 2, raw_vals, 1, nraw)
    @inbounds vals[nraw + 2] = raw_vals[nraw]
    PixelLUT(vals, lowerbnd)
end

@inline function getindex(a::PixelLUT, i)
    i_raw = i - a.lowerbnd + 2
    i_clamp = min(max(i_raw, 1), length(a.vals))
    @inbounds a.vals[i_clamp]
end

@inline getindex(a::PixelLUT, i::Normed) = a[reinterpret(i)]

setindex!(::PixelLUT, ::Any) = throw(ReadOnlyMemoryError())
size(a::PixelLUT) = (length(a.vals) - 2,)
IndexStyle(::Type{PixelLUT}) = IndexLinear()
axes(a::PixelLUT) =
    (IdentityUnitRange(a.lowerbnd:a.lowerbnd + length(a.vals) - 3),)

function _apply_lut!(lut, dest, src, lo, hi)
    @inbounds @simd ivdep for elno in lo:hi
        dest[elno] = lut[src[elno]]
    end
end

function apply_lut!(lut::PixelLUT, dest, src; nt = Threads.nthreads())
    nel = length(src)
    nel == length(dest) || throw(ArgumentError("inputs not the same size"))
    if nt > 1
        blksize = cld(nel, nt)
        tasks = Vector{Task}(undef, nt)
        for tno in 1:nt
            lo = (tno - 1) * blksize + 1
            hi = min(tno * blksize, nel)
            @inbounds tasks[tno] = @spawn _apply_lut!(lut, dest, src, lo, hi)
        end
        foreach(wait, tasks)
    else
        _apply_lut!(lut, dest, src, 1, nel)
    end
    return dest
end

apply_lut!(lut, src; kwargs...) = apply_lut(lut, src, src; kwargs...)

apply_lut(lut, arr; kwargs...) = apply_lut!(lut, similar(arr), arr; kwargs...)

function frame_min(imgs::AbstractArray{T, 3}) where T
    nr, nc, nf = size(imgs)
    reduced_frames = reduce(min, imgs; dims = (3,), init = typemax(T))
    return reshape(reduced_frames, (nr, nc))
end

function _frame_min_max!(minframe, maxframe, thisframe, rowrange, lo, hi)
    for colno in lo:hi
        @inbounds @simd ivdep for rowno in rowrange
            thisval = thisframe[rowno, colno]
            minframe[rowno, colno] = min(minframe[rowno, colno], thisval)
            maxframe[rowno, colno] = max(maxframe[rowno, colno], thisval)
        end
    end
end

function frame_min_max!(minframe, maxframe, tasks, thisframe, rowrange, los,
                        his)
    @inbounds for tno in eachindex(tasks)
        tasks[tno] = @spawn _frame_min_max!(minframe, maxframe, thisframe,
                                            rowrange, los[tno], his[tno])
    end
    foreach(wait, tasks)
    return minframe, maxframe
end

function frame_min_max!(minframe, maxframe, thisframe; nt = Threads.nthreads())
    nr, nc = size(thisframe)
    rowrange = 1:nr
    if nt > 1
        blksize = cld(nc, nt)
        tasks = Vector{Task}(undef, nt)
        for tno in 1:nt
            lo = (tno - 1) * blksize + 1
            hi = min(tno * blksize, nc)
            @inbounds tasks[tno] = @spawn _frame_min_max!(minframe, maxframe,
                                                          thisframe, rowrange,
                                                          lo, hi)
        end
        foreach(wait, tasks)
    else
        _frame_min_max!(minframe, maxframe, thisframe, rowrange, 1, nc)
    end
    return minframe, maxframe
end

function frames_min_max!(minframe, maxframe, imgs::AbstractArray{<:Any, 3};
                        nt = Threads.nthreads())
    nr, nc, nf = size(imgs)
    rowrange = 1:nr
    if nt > 1
        blksize = cld(nc, nt)
        tasks = Vector{Task}(undef, nt)
        los = Vector{Int}(undef, nt)
        his = similar(los)
        @inbounds for tno in 1:nt
            los[tno] = (tno - 1) * blksize + 1
            his[tno] = min(tno * blksize, nc)
        end
        tasks = Vector{Task}(undef, nt)
        for fno in 1:nf
            frame_min_max!(minframe, maxframe, tasks, view(imgs, :, :, fno),
                           rowrange, los, his)
        end
    else
        for fno in 1:nf
            _frame_min_max!(minframe, maxframe, view(imgs, :, :, fno), rowrange,
                            1, nc)
        end
    end
    return minframe, maxframe
end

function frames_min_max(imgs::AbstractArray{T, 3}; nt = Threads.nthreads()) where T
    nr, nc, nz = size(imgs)
    minframe = fill(typemax(T), (nr, nc))
    maxframe = fill(typemin(T), (nr, nc))
    frames_min_max!(minframe, maxframe, imgs, nt = nt)
end

function _subtract_frame!(dest, thisframe, rmframe, rowrange, lo, hi)
    for cno in lo:hi
        @inbounds @simd ivdep for rno in rowrange
            dest[rno, cno] = thisframe[rno, cno] - rmframe[rno, cno]
        end
    end
end

function subtract_frame!(dest, thisframe, rmframe; nt = Threads.nthreads())
    sz = size(thisframe)
    size(dest) == sz || throw(ArgumentError("Sizes are not the same"))
    nr, nc = sz
    rowrange = 1:nr
    if nt > 1
        blksize = cld(nc, nt)
        tasks = Vector{Task}(undef, nt)
        for tno in 1:nt
            lo = (tno - 1) * blksize + 1
            hi = min(tno * blksize, nc)
            @inbounds tasks[tno] = @spawn _subtract_frame!(dest, thisframe,
                                                           rmframe, rowrange,
                                                           lo, hi)
        end
        foreach(wait, tasks)
    else
        _subtract_frame!(dest, thisframe, rmframe, rowrange, 1, nc)
    end
    return dest
end

function subtract_frame!(dest, tasks, thisframe, rmframe, rowrange, los, his)
    @inbounds for tno in eachindex(tasks)
        tasks[tno] = @spawn _subtract_frame!(dest, thisframe, rmframe, rowrange,
                                             los[tno], his[tno])
    end
    foreach(wait, tasks)
    return dest
end

subtract_frame(frame, rmframe; kwargs...) =
    subtract_frame!(similar(frame), frame, rmframe; kwargs...)

function subtract_frames!(dest, frames, rmframe; nt = Threads.nthreads())
    sz = size(frames)
    size(dest) == sz || throw(ArgumentError("Sizes are not the same"))
    nr, nc, nf = sz
    rowrange = 1:nr
    if nt > 1
        blksize = cld(nc, nt)
        tasks = Vector{Task}(undef, nt)
        los = similar(tasks, Int)
        his = similar(los)
        @inbounds for tno in 1:nt
            los[tno] = (tno - 1) * blksize + 1
            his[tno] = min(tno * blksize, nc)
        end
        @inbounds for fno in 1:nf
            subtract_frame!(view(dest, :, :, fno), tasks,
                            view(frames, :, :, fno), rmframe, rowrange, los,
                            his)
        end
    else
        @inbounds for fno in 1:nf
            _subtract_frame!(view(dest, :, :, fno), view(frames, :, :, fno),
                             rmframe, rowrange, 1, nc)
        end
    end
    return dest
end

subtract_frames(frames, rmframe; kwargs...) =
    subtract_frames!(similar(frames), frames, rmframe; kwargs...)

function demin!(dest::AbstractArray, imgs::AbstractArray; nt = Threads.nthreads())
    size(dest) == size(imgs) || throw(ArgumentError("Sizes are not the same"))
    minframe = frame_min(imgs)
    subtract_frames!(dest, imgs, minframe, nt = nt)
    return dest
end

demin!(imgs; kwargs...) = demin!(imgs, imgs; kwargs...)

function demin(imgs; parentdir = tempdir(), kwargs...)
    if !isempty(parentdir) && isdir(parentdir)
        mpath, mio = mktemp(parentdir)
        dest = Mmap.mmap(mio, Array{eltype(imgs), 3}, size(imgs))
        close(mio)
        rm(mpath)
    else
        dest = similar(imgs)
    end
    demin!(dest, imgs; kwargs...)
end

function _avg_intensities!(intensities, imgs, lo, hi)
    @inbounds @simd for fno in lo:hi
        intensities[fno] = mean(imgs[:, :, fno])
    end
end

function avg_intensities(imgs::AbstractArray{<:Any, 3}; nt = Threads.nthreads())
    nf = size(imgs, 3)
    intensities = Vector{Float64}(undef, nf)
    if nt > 1
        blksize = cld(nf, nt)
        tasks = Vector{Task}(undef, nt)
        for tno in 1:nt
            lo = (tno - 1) * blksize + 1
            hi = min(tno * blksize, nf)
            @inbounds tasks[tno] = @spawn _avg_intensities!(intensities, imgs, lo, hi)
        end
        foreach(wait, tasks)
    else
        _avg_intensities!(intensities, imgs, 1, nf)
    end
    return intensities
end

function _max_intensities!(intensities, imgs, lo, hi)
    @inbounds @simd for fno in lo:hi
        intensities[fno] = maximum(imgs[:, :, fno])
    end
end

function max_intensities(imgs::AbstractArray{<:Any, 3}; nt = Threads.nthreads())
    nf = size(imgs, 3)
    intensities = similar(imgs, (nf,))
    if nt > 1
        blksize = cld(nf, nt)
        tasks = Vector{Task}(undef, nt)
        for tno in 1:nt
            lo = (tno - 1) * blksize + 1
            hi = min(tno * blksize, nf)
            @inbounds tasks[tno] = @spawn _max_intensities!(intensities, imgs, lo, hi)
        end
        foreach(wait, tasks)
    else
        _max_intensities!(intensities, imgs, 1, nf)
    end
    return intensities
end

function clip_imgs(imgs; x = :, y = :)
    nf = size(imgs, 3)
    view(imgs, x, y, 1:nf)
end

function find_segments_thr(imgs, thr; nt = Threads.nthreads())
    intensities = avg_intensities(imgs, nt = nt)
    indices_above_thresh(intensities, thr)
end

"""
Assumes images are row-major
"""
function clip_segments_thr(imgs, thr; nt = Threads.nthreads())
    open_pers = find_segments_thr(imgs, thr, nt = nt)
    nseg = length(open_pers)
    segments = map(1:nseg) do segno
        view(imgs, :, :, open_pers[segno])
    end
    return segments, open_pers
end

function map_to_8bit(imgs::AbstractArray{<:Normed, 3})
    if reinterpret(UInt16, maximum(imgs)) > typemax(UInt8)
        throw(ArgumentError("Maximum exceeds 8 bits"))
    end
    out = similar(imgs, N0f8)
    outcnts = rawview(out)
    cnts = rawview(imgs)
    copyto!(outcnts, 1, cnts, 1, length(cnts))
    return out
end

srgb_gamma_compress(x) = x <= 0.0031308 ?
    323 * x / 25 :
    (211 * x ^ (5 / 12) - 11) / 200

srgb_gamma_expand(x) = x <= 0.04045 ?
    25 * x / 323 :
    ((200 * x + 11) / 211)^(12 / 5)

@inline function rescale_compress(::Type{UInt8}, x::Integer, scale::Float64)
    scaled_val = srgb_gamma_compress(scale * x)
    return round(UInt8, typemax(UInt8) * scaled_val)
end

rescale_compress(d::DataType, x::Normed, scale::Float64) =
    rescale_compress(d, reinterpret(x), scale)

function __rescale_compress_img!(outimg::AbstractMatrix{T}, inimg, lut, lo,
                                 hi) where T
    rows = 1:size(inimg, 1)
    @inbounds for colno in lo:hi, rowno in rows
        outimg[rowno, colno] = lut[inimg[rowno, colno]]
    end
end

function _rescale_compress_img!(outimg::AbstractMatrix, inimg,
                                lut; nt = 4)
    ncol  = size(inimg, 2)
    if nt > 1
        tasks = Vector{Task}(undef, nt)
        blksize = cld(ncol, nt)
        for tno in 1:nt
            lo = (tno - 1) * blksize + 1
            hi = min(tno * blksize, ncol)
            @inbounds tasks[tno] = @spawn __rescale_compress_img!(
                outimg, inimg, lut, lo, hi
            )
        end
        foreach(wait, tasks)
    else
        __rescale_compress_img!(outimg, inimg, lut, 1, ncol)
    end
    outimg
end

function rescale_compress_img!(outimg, inimg, lut; kwargs...)
    if size(outimg) != size(inimg)
        throw(ArgumentError("outimg and inimg must have the same size"))
    end
    _rescale_compress_img!(outimg, inimg, lut; kwargs...)
end

function rescale_compress_img(inimg, lut)
    rescale_compress_img!(similar(inimg, UInt8), inimg, lut)
end

end # module
