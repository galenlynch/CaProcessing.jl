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

function frame_min(imgs::AbstractArray{T, 3}) where T
    nr, nc, nf = size(imgs)
    reduced_frames = reduce(min, imgs; dims = (3,), init = typemax(T))
    return reshape(reduced_frames, (nr, nc))
end

function _demin!(dest, imgs, minframe, lo, hi)
    @inbounds @simd for fno in lo:hi
        dest[:, :, fno] .= imgs[:, :, fno] .- minframe
    end
end

function demin!(dest::AbstractArray, imgs::AbstractArray; nt = Threads.nthreads())
    sz = size(imgs)
    size(dest) == sz || throw(ArgumentError("Sizes are not the same"))
    nf = sz[3]
    minframe = frame_min(imgs)
    if nt > 1
        blksize = cld(nf, nt)
        tasks = Vector{Task}(undef, nt)
        for tno in 1:nt
            lo = (tno - 1) * blksize + 1
            hi = min(tno * blksize, nf)
            @inbounds tasks[tno] = @spawn _demin!(dest, imgs, minframe, lo, hi)
        end
        foreach(wait, tasks)
    else
        _demin!(dest, imgs, minframe, 1, nf)
    end
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
