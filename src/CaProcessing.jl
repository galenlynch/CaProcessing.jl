module CaProcessing

using Base.Threads: @spawn, nthreads

import Base: getindex, firstindex, lastindex, size, setindex!, IndexStyle,
    axes, axes1, IdentityUnitRange, Slice

# Stdlib
using Mmap, Statistics
# Public packages
using FixedPointNumbers: Normed, N0f8, N6f10, floattype
using ImageCore: rawview
using ImageTransformations: restrict
# Private packages
using GLUtilities: indices_above_thresh, reduce_extrema, clip_interval_duration

export pixel_lut,
    apply_lut,
    apply_lut!,
    avg_intensities,
    clip_imgs,
    clip_segments_thr,
    demin,
    demin!,
    find_segments_thr,
    find_mean_and_scale_fun,
    frame_min_max!,
    frames_min_max,
    frames_min_max!,
    frames_min_max_mean,
    frames_min_max_accum,
    gamma_compensate_rescale,
    map_to_8bit,
    make_scale_f,
    make_pixel_lut,
    max_intensities,
    srgb_gamma_compress,
    srgb_gamma_expand,
    rescale_brightness,
    rescale_clamp_brightness,
    rescale_replace_brightness,
    rescale_compress,
    rescale_compress_img,
    rescale_compress_img!,
    subtract_frame,
    subtract_frame!,
    subtract_frames

include("median_filter.jl")

struct PixelLUT{T} <: AbstractArray{T,1}
    vals::Vector{T}
    lowerbnd::Int
end
PixelLUT(vals, lowerbnd::Integer) = PixelLUT(vals, convert(Int, lowerbnd))
PixelLUT(vals, lowerbnd::Normed) = PixelLUT(vals, reinterpret(lowerbnd))

"""
    pixel_lut(f, ::Type{T}, r::AbstractUnitRange, lowval, hival) where T

Construct a [`PixelLUT`](@ref) by applying `f` to each element of `r`, and
converting the result to type `T`. Values below and above `r` will be mapped
onto `lowval` and `highval`, respectively.
"""
function pixel_lut(f, ::Type{T}, r::AbstractRange, lowval, hival) where T
    vals = Vector{T}(undef, length(r) + 2)
    @inbounds vals[1] = lowval
    @inbounds for (i, x) in enumerate(r)
        vals[i + 1] = f(x)
    end
    @inbounds vals[end] = hival
    PixelLUT(vals, first(r))
end

function pixel_lut(f, r::AbstractRange)
    isempty(r) &&
        throw(ArgumentError("Must specify type and end values if r is empty"))
    lowval = f(first(r))
    hival = f(last(r))
    T = typeof(lowval)
    pixel_lut(f, T, r, lowval, hival)
end

pixel_lut(f, i::T) where T<:Integer = pixel_lut(f, zero(T):i)
pixel_lut(f, i::T) where T<:Normed = pixel_lut(f, zero(T):eps(T):i)

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
    i_clamp = clamp(i_raw, 1, length(a.vals))
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

function apply_lut!(lut::PixelLUT, dest, src; nt = nthreads())
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

apply_lut!(lut, src; kwargs...) = apply_lut!(lut, src, src; kwargs...)

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

function frame_min_max!(minframe, maxframe, thisframe; nt = nthreads())
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
                        nt = nthreads())
    nr, nc, nf = size(imgs)
    rowrange = 1:nr
    if nt > 1
        blksize = cld(nc, nt)
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

function frames_min_max(imgs::AbstractArray{T, 3}; nt = nthreads()) where T
    nr, nc, nz = size(imgs)
    minframe = fill(typemax(T), (nr, nc))
    maxframe = fill(typemin(T), (nr, nc))
    frames_min_max!(minframe, maxframe, imgs, nt = nt)
end

function _frame_min_max_accum!(minf, maxf, accf, thisframe, rowrange, colrange)
    for colno in colrange
        @inbounds @simd ivdep for rowno in rowrange
            thisval = thisframe[rowno, colno]
            minf[rowno, colno] = min(minf[rowno, colno], thisval)
            maxf[rowno, colno] = max(maxf[rowno, colno], thisval)
            accf[rowno, colno] = accf[rowno, colno] + thisval
        end
    end
end

function frame_min_max_accum!(minf, maxf, accf, tasks, thisframe, rowrange,
                              colranges)
    if isempty(tasks)
        _frame_min_max_accum!(minf, maxf, accf, thisframe, rowrange,
                              colranges[1])
    else
        @inbounds for tno in eachindex(tasks)
            tasks[tno] = @spawn _frame_min_max_accum!(minf, maxf, accf, thisframe,
                                                      rowrange, colranges[tno])
        end
        foreach(wait, tasks)
    end
    return minf, maxf, accf
end

function frame_min_max_accum!(minf, maxf, accf, thisframe; nt = nthreads())
    nr, nc = size(thisframe)
    rowrange = 1:nr
    if nt > 1
        blksize = cld(nc, nt)
        tasks = Vector{Task}(undef, nt)
        for tno in 1:nt
            lo = (tno - 1) * blksize + 1
            hi = min(tno * blksize, nc)
            @inbounds tasks[tno] = @spawn _frame_min_max_accum!(minf, maxf, accf,
                                                          thisframe, rowrange,
                                                          lo:hi)
        end
        foreach(wait, tasks)
    else
        _frame_min_max_accum!(minf, maxf, accf, thisframe, rowrange, 1, nc)
    end
    return minf, maxf, accf
end

function frames_min_max_accum!(minf, maxf, accf, imgs; nt = nthreads())
    nr, nc, nf = size(imgs)
    rowrange = 1:nr
    if nt > 1
        blksize = cld(nc, nt)
        colranges = Vector{UnitRange{Int}}(undef, nt)
        @inbounds for tno in 1:nt
            colranges[tno] = (tno - 1) * blksize + 1 : min(tno * blksize, nc)
        end
        tasks = Vector{Task}(undef, nt)
        for fno in 1:nf
            frame_min_max_accum!(minf, maxf, accf, tasks, view(imgs, :, :, fno),
                                rowrange, colranges)
        end
    else
        colrange = 1:nc
        for fno in 1:nf
            _frame_min_max_accum!(minf, maxf, accf, view(imgs, :, :, fno),
                                  rowrange, colrange)
        end
    end
    minf, maxf, accf
end

function frames_min_max_accum_alloc(::Type{S}, ::Type{T}, sz) where {S, T}
    minf = Array{T}(undef, sz)
    maxf = similar(minf)
    accf = Array{S}(undef, sz)
    minf, maxf, accf
end

function frames_min_max_accum_init!(minf::AbstractArray{T},
                                    maxf::AbstractArray{T},
                                    accf) where T
    fill!(minf, typemax(T))
    fill!(maxf, typemin(T))
    fill!(accf, 0)
    minf, maxf, accf
end

function frames_min_max_accum(::Type{S}, imgs::AbstractArray{T, 3};
                            kwargs...) where {S,T}
    nr, nc, nz = size(imgs)
    minf, maxf, accf = frames_min_max_accum_alloc(S, T, (nr, nc))
    frames_min_max_accum_init!(minf, maxf, accf)
    frames_min_max_accum!(minf, maxf, accf, imgs; kwargs...)
end
frames_min_max_accum(imgs; kwargs...) = frames_min_max_accum(UInt32, imgs; kwargs...)

function frames_min_max_mean(::Type{S}, ::Type{T}, imgs; kwargs...) where {S,T}
    nf = size(imgs, 3)
    nf > 0 || throw(ArugmentError("imgs must not be empty"))
    minf, maxf, accf = frames_min_max_accum(T, imgs; kwargs...)
    meanf = similar(accf, S)
    meanf .= accf ./ nf
    minf, maxf, meanf
end
frames_min_max_mean(imgs::AbstractArray{<:Integer}; kwargs...) =
    frames_min_max_mean(Float32, UInt32, imgs; kwargs...)
frames_min_max_mean(imgs::AbstractArray{<:AbstractFloat}; kwargs...) =
    frames_min_max_mean(Float32, Float64, imgs; kwargs...)


function find_mean_and_scale_fun(::Type{T}, signal, newmax, usegamma = false
                                 ) where T
    minf, maxf, meanf = frames_min_max_mean(signal)
    minv = typemax(eltype(meanf))
    maxv = typemin(minv)
    @inbounds for i in eachindex(meanf)
        meanval = meanf[i]
        minv = min(minv, minf[i] - meanval)
        maxv = max(maxv, maxf[i] - meanval)
    end
    f = make_scale_f(T, minv, maxv, newmax, usegamma)
    f, meanf
end

function _subtract_frame!(dest, thisframe, rmframe, rowrange, lo, hi)
    for cno in lo:hi
        @inbounds @simd ivdep for rno in rowrange
            dest[rno, cno] = thisframe[rno, cno] - rmframe[rno, cno]
        end
    end
end

function subtract_frame!(dest, thisframe, rmframe; nt = nthreads())
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

function subtract_frame(frame::AbstractArray{T},
                        rmframe::AbstractArray{S}; kwargs...) where {S,T}
    subtract_frame!(similar(frame, promote_type(T, S)), frame, rmframe;
                    kwargs...)
end

function subtract_frames!(dest, frames, rmframe; nt = nthreads())
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

function demin!(dest::AbstractArray, imgs::AbstractArray; nt = nthreads())
    size(dest) == size(imgs) || throw(ArgumentError("Sizes are not the same"))
    minframe = frame_min(imgs)
    subtract_frames!(dest, imgs, minframe, nt = nt)
    return dest
end

demin!(imgs; kwargs...) = demin!(imgs, imgs; kwargs...)

function demin(imgs; scratch_dir = tempdir(), kwargs...)
    if !isempty(scratch_dir) && isdir(scratch_dir)
        mpath, mio = mktemp(scratch_dir)
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

function avg_intensities(imgs::AbstractArray{<:Any, 3}; nt = nthreads())
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

function max_intensities(imgs::AbstractArray{<:Any, 3}; nt = nthreads())
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

function find_segments_thr(imgs, thr; nt = nthreads())
    intensities = avg_intensities(imgs, nt = nt)
    indices_above_thresh(intensities, thr)
end

"""
Assumes images are row-major
"""
function clip_segments_thr(imgs, thr; nt = nthreads())
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

rawval(x::Normed) = reinterpret(x)
rawval(x) = x

rescale_brightness(::Type{T}, x, xmin, xmax, newmax) where T<:Integer =
    round(T, newmax * float(x - xmin) / (xmax - xmin))
rescale_brightness(::Type{T}, x, xmin, xmax, newmax) where T =
    convert(T, newmax * float(x - xmin) / (xmax - xmin))

rescale_brightness(::Type{T}, x::T, xmin, xmax, newmax) where {X, T<:Normed{X}} =
    reinterpret(T, rescale_brightness(X, rawval(x), rawval(xmin),
                                      rawval(xmax), rawval(newmax)))

rescale_brightness(x::T, xmin, xmax, newmax) where T =
    rescale_brightness(T, x, xmin, xmax, newmax)

rescale_brightness(x, xmax, newmax) = rescale_brightness(x, 0, xmax, newmax)

rescale_clamp_brightness(::Type{T}, x, b, e, newmax) where T =
    rescale_brightness(T, clamp(x, b, e), b, e, newmax)

function rescale_replace_brightness(::Type{T}, x, b, e, fillval,
                                    newmax = typemax(T)) where T
    rescale_brightness(T, ifelse(b <= x <= e, x, fillval), b, e, newmax)
end

scale_clamp_f(::Type{T}, b, e, m = typemax(T)) where T =
    x -> rescale_brightness(T, clamp(x, b, e), b, e, m)


function gamma_compensate_rescale(::Type{T}, x, xmin, xmax, newmax) where T <: Integer
    gamma_float = srgb_gamma_compress(rescale_brightness(Float64, x, xmin, xmax, 1))
    return round(T, newmax * gamma_float)
end
function gamma_compensate_rescale(::Type{T}, x, xmin, xmax, newmax) where T
    gamma_float = srgb_gamma_compress(rescale_brightness(Float64, x, xmin, xmax, 1))
    convert(T, gamma_float)
end
gamma_compensate_rescale(x::T, args...) where T = gamma_compensate_rescale(T, x, args...)

px_step_range(b::T, e::T) where T = b:e
px_step_range(b::T, e::T) where T<:Normed = b:eps(T):e
px_step_range(b, e) = px_step_range(promote(b, e)...)

function make_pixel_lut(::Type{T}, minval, maxval, newmax, use_gamma = false) where T
    r = px_step_range(minval, maxval)
    if use_gamma
        lut = pixel_lut(x -> gamma_compensate_rescale(T, x, minval, maxval,
                                                      newmax), r)
    else
        lut = pixel_lut(x -> rescale_brightness(T, x, minval, maxval, newmax),
                        r)
    end
    lut
end

make_pixel_lut(minv::T, maxv::T, newmax, args...) where T =
    make_pixel_lut(T, minv, maxv, newmax, args...)

function make_scale_f(::Type{T}, minval, maxval, newmax, use_gamma = false) where T
    if use_gamma
        scale_f = x -> gamma_compensate_rescale(T, x, minval, maxval, newmax)
    else
        scale_f = x -> rescale_brightness(T, x, minval, maxval, newmax)
    end
    scale_f
end

make_scale_f(minv::T, maxv::T, newmax, args...) where T =
    make_scale_f(T, minv, maxv, newmax, args...)

@inline function rescale_compress(::Type{UInt8}, x::Integer, scale::Float64)
    scaled_val = srgb_gamma_compress(scale * x)
    return round(UInt8, typemax(UInt8) * scaled_val)
end

@inline function rescale_compress(::Type{T}, x::Normed, scale::AbstractFloat) where T<:Normed
    scaled_val = srgb_gamma_compress(scale * x)
    convert(T, scaled_val)
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

lut_maxrange(minv, maxv, newmax) = make_pixel_lut(minv, maxv, newmax)
lut_maxrange(minv::T, maxv::T) where T<:Integer =
    lut_maxrange(minv, maxv, typemax(T))
lut_maxrange(minval::T, maxval::T) where T<:Normed =
    lut_maxrange(minval, maxval, one(T))
lut_maxrange(minv, maxv, ::Nothing) = lut_maxrange(minv, maxv)

function lut_maxrange(imgstack, newmax = nothing)
    minval, maxval = mapreduce(extrema, reduce_extrema, imgstack)
    lut_maxrange(minval, maxval, newmax)
end

function lut_maxrange(imgstack::AbstractArray{T, 3}, newmax = nothing) where T
    minval, maxval = extrema(imgstack)
    lut_maxrange(minval, maxval, newmax)
end

function determine_container_depth(maxval::N6f10)
    maxval_8bit = reinterpret(N6f10, convert(UInt16, reinterpret(one(N0f8))))
    maxval <= maxval_8bit ? N0f8 : N6f10
end

determine_container_depth(maxval) = maxval <= typemax(UInt8) ? N0f8 : N6f10

larger_container_type(::Type{T}, ::Type{T}) where T = T
larger_container_type(::Type{Base.Bottom}, ::Type{T}) where T = T
larger_container_type(::Type{T}, ::Type{Base.Bottom}) where T = T
larger_container_type(::Type{Base.Bottom}, ::Type{Base.Bottom}) = Base.Bottom
function larger_container_type(::Type{S}, ::Type{T}) where {S,T}
    larger_container_type(larger_container_rule(S, T), larger_container_rule(T, S))
end

larger_container_rule(::Type{<:Any}, ::Type{<:Any}) = Base.Bottom
function larger_container_rule(::Type{X}, ::Type{Y}) where {A, f, X<:Normed{A, f},
                                                            B, g, Y<:Normed{B, g}}
    if f < g
        T = Y
    elseif f == g
        # Prefer smaller containers with the same bit depth
        T = sizeof(A) < sizeof(B) ? X : Y
    else
        T = X
    end
    T
end

function max_container_depth(maxvals)
    mapreduce(determine_container_depth, larger_container_type,
              maxvals, init = N0f8)
end

function _frame_sum_intensity(f, ::Type{T}, img_raw, roi_xr, roi_yr) where T
    accum = zero(T)
    @inbounds for yi in eachindex(roi_yr)
        for xi in eachindex(roi_xr)
            intensity = f(img_raw[roi_xr[xi], roi_yr[yi]])
            accum += intensity
        end
    end
    accum
end

function frame_sum_intensity(f, ::Type{T}, img_raw::AbstractArray;
                             roi_xr = 1:size(img_raw, 1),
                             roi_yr = 1:size(img_raw, 2), nt = nthreads()) where T
    if nt > 1
        ny = length(roi_yr)
        blksize = cld(ny, nt)
        tasks = Vector{Task}(undef, nt)
        @inbounds for tno in 1:nt
            lo = (tno - 1) * blksize + first(roi_yr)
            hi = min(first(roi_yr) + tno * blksize - 1, last(roi_yr))
            tasks[tno] = @spawn _frame_sum_intensity(f, T, img_raw, roi_xr, lo:hi)
        end
        accum = zero(T)
        @inbounds for tno in 1:nt
            accum += fetch(tasks[tno])::T
        end
    else
        accum = _frame_sum_intensity(f, T, img_raw, roi_xr, roi_yr)
    end
    accum
end

function frame_sum_intensity(f::typeof(identity),
                             img_raw::AbstractArray{UInt16}; kwargs...)
    frame_sum_intensity(f, UInt64, img_raw; kwargs...)
end

function frame_sum_intensity(f::typeof(identity), img_raw::AbstractArray{T};
                             kwargs...) where {X, T<:Normed{X}}
    frame_sum_intensity(f, reinterpret(X, img_raw); kwargs...)
end

frame_sum_intensity(img_raw; kwargs...) = frame_sum_intensity(identity, img_raw; kwargs...)

frame_avg_intensity(f, ::Type{T}, img_raw::AbstractArray, norm;
                    kwargs...) where T =
    norm * frame_sum_intensity(f, T, img_raw; kwargs...)

function frame_avg_intensity(f, ::Type{T}, img_raw; roi_xr = 1:size(img_raw, 1),
                             roi_yr = 1:size(img_raw, 2), kwargs...) where T
    norm = get_norm(roi_xr, roi_yr)
    frame_avg_intensity(f, T, img_raw, norm; roi_xr, roi_yr, kwargs...)
end

frame_avg_intensity(::Type{T}, img_raw::AbstractArray, args...;
                    kwargs...) where T =
    frame_avg_intensity(identity, T, img_raw, args...; kwargs...)
frame_avg_intensity(img_raw::AbstractArray, args...; kwargs...) =
    frame_avg_intensity(UInt64, img_raw, args...; kwargs...)

function get_norm(nx::Integer, ny::Integer)
    npx = nx * ny
    1 / npx
end

get_norm(roi_x::AbstractUnitRange, roi_y::AbstractUnitRange) =
    get_norm(length(roi_x), length(roi_y))
get_norm(roi_x::Slice, roi_y::Slice) = get_norm(length(roi_x), length(roi_y))
get_norm(img::AbstractMatrix) = get_norm(size(img)...)

function _maxval_min_max_frames(roi_min, roi_max, ib, ie)
    subtr_mv = typemin(eltype(roi_min))
    @inbounds @simd ivdep for i in ib:ie
        subtr_mv = max(roi_max[i] - roi_min[i], subtr_mv)
    end
    subtr_mv
end

function maxval_min_max_frames(roi_min::AbstractArray{T}, roi_max, nt = nthreads()) where T
    nel = length(roi_min)
    if nt > 1
        blksize = cld(nel, nt)
        tasks = Vector{Task}(undef, nt)
        for tno in 1:nt
            lo = (tno - 1) * blksize + 1
            hi = min(tno * blksize, nel)
            @inbounds tasks[tno] = @spawn _maxval_min_max_frames(roi_min,
                                                                 roi_max, lo, hi)
        end
        subtr_mv = typemin(T)
        for tno in 1:nt
            this_mv = fetch(tasks[tno])::T
            subtr_mv = max(this_mv, subtr_mv)
        end
    else
        subtr_mv = _maxval_min_max_frames(roi_min, roi_max, 1, nel)
    end
    subtr_mv
end

restrict_size(n) = iseven(n) ? div(n, 2) + 1 : div(n + 1, 2)

function restrict_no_edges!(dest, img::AbstractMatrix)
    halved = restrict(img)
    dest .= halved[2 : end - 1, 2 : end - 1]
    dest
end

restrict_no_edges(img::AbstractArray{T}) where T =
    restrict_no_edges!(similar(img, floattype(T), restrict_size.(size(img)) .- 2),
                       img)

function _frame_pixel_map!(pixel_f::T, dest::AbstractMatrix{<:Any},
                           stack::AbstractArray{<:Any, 3}, fno, colrange,
                           rowrange, ref_frame::AbstractMatrix) where T
    for c in colrange
        @inbounds @simd for r in rowrange
            dest[r, c] = pixel_f(stack[r, c, fno], ref_frame[r, c])
        end
    end
    nothing
end

function _frame_pixel_map!(pixel_f::T, dest::AbstractMatrix{<:Any},
                           stack::AbstractArray{<:Any, 3}, fno, colrange,
                           rowrange) where T
    for c in colrange
        @inbounds @simd for r in rowrange
            dest[r, c] = pixel_f(stack[r, c, fno])
        end
    end
    nothing
end

function frame_map!(pixel_f::T, dest::AbstractMatrix{<:Any},
                    stack::AbstractArray{<:Any, 3}, args...;
                    nt = nthreads()) where {S, T}
    nr, nc, nf = size(stack)
    rowrange = 1:nr
    if nt > 1
        colranges = splits(1:nc, nt)
        tasks = Vector{Task}(undef, nt)
        for fno in 1:nf
            @inbounds for tno in 1:nt
                tasks[tno] = @spawn _frame_pixel_map!(pixel_f, dest, stack, fno,
                                                      colranges[tno], rowrange,
                                                      dest)
            end
            foreach(wait, tasks)
        end
    else
        colrange = 1:nc
        for fno in 1:nf
            _frame_pixel_map!(pixelf, dest, stack, fno, colrange, rowrange, dest)
        end
    end
    dest
end

function frame_sink_map!(frame_sink_f::S, pixel_f::T,
                         dest::AbstractMatrix{<:Any},
                         stack::AbstractArray{<:Any, 3}, args...;
                         nt = nthreads()) where {S, T}
    nr, nc, nf = size(stack)
    rowrange = 1:nr
    if nt > 1
        colranges = splits(1:nc, nt)
        tasks = Vector{Task}(undef, nt)
        for fno in 1:nf
            @inbounds for tno in 1:nt
                tasks[tno] = @spawn _frame_pixel_map!(pixel_f, dest, stack, fno,
                                                      colranges[tno], rowrange,
                                                      args...)
            end
            foreach(wait, tasks)
            frame_sink_f(dest, fno)
        end
    else
        colrange = 1:nc
        for fno in 1:nf
            _frame_pixel_map!(pixelf, dest, stack, fno, colrange, rowrange,
                              args...)
            frame_sink_f(dest, fno)
        end
    end
    nothing
end

function _stack_map!(pixel_f::T, dest::AbstractArray{<:Any, 3},
                     stack::AbstractArray{<:Any, 3}, fno, colrange,
                     rowrange, ref_frame::AbstractMatrix) where T
    for c in colrange
        @inbounds @simd for r in rowrange
            dest[r, c, fno] = pixel_f(stack[r, c, fno], ref_frame[r, c])
        end
    end
    nothing
end

function _stack_map!(pixel_f::T, dest::AbstractArray{<:Any, 3},
                     stack::AbstractArray{<:Any, 3}, fno, colrange,
                     rowrange) where T
    for c in colrange
        @inbounds @simd for r in rowrange
            dest[r, c, fno] = pixel_f(stack[r, c, fno])
        end
    end
    nothing
end

function stack_map!(pixel_f::T, dest::AbstractArray{<:Any, 3},
                    stack::AbstractArray{<:Any, 3}, args...; nt = nthreads()) where T
    nr, nc, nf = size(stack)
    rowrange = 1:nr
    if nt > 1
        colranges = splits(1:nc, nt)
        tasks = Vector{Task}(undef, nt)
        for fno in 1:nf
            @inbounds for tno in 1:nt
                tasks[tno] = @spawn _stack_map!(pixel_f, dest, stack, fno,
                                                colranges[tno], rowrange,
                                                args...)
            end
            foreach(wait, tasks)
        end
    else
        colrange = 1:nc
        for fno in 1:nf
            _stack_map!(pixelf, dest, stack, fno, colrange, rowrange, args...)
        end
    end
    nothing

end

function __stack_temporal_downsample_accum!(ds, stack, dstframe, srcframe,
                                            colrange, rowrange)
    for c in colrange
        @inbounds @simd for r in rowrange
            ds[r, c, dstframe] += stack[r, c, srcframe]
        end
    end
end

function __stack_temporal_downsample_scale!(ds, factor, fno, colrange, rowrange)
    for c in colrange
        @inbounds @simd for r in rowrange
            ds[r, c, fno] /= factor
        end
    end
end

function _stack_temporal_downsample!(ds, stack, factor; nt = nthreads())
    nr, nc, nff = size(ds)
    rowrange = 1:nr
    if nt > 1
        colranges = splits(1:nc, nt)
        tasks = Vector{Task}(undef, nt)
        for i in 1:nff
            for j in 1:factor
                srcframe = factor * (i - 1) + j
                @inbounds for tno in 1:nt
                    tasks[tno] = @spawn __stack_temporal_downsample_accum!(
                        ds, stack, i, srcframe, colranges[tno], rowrange
                    )
                end
                foreach(wait, tasks)
            end
            @inbounds for tno in 1:nt
                tasks[tno] = @spawn __stack_temporal_downsample_scale!(
                    ds, factor, i, colranges[tno], rowrange
                )
            end
            foreach(wait, tasks)
        end
    else
        colrange = 1:nc
        for i in 1:nff
            for j in 1:factor
                srcframe = factor * (i - 1) + j
                __stack_temporal_downsample_accum!(ds, stack, i, srcframe,
                                                   colrange, rowrange)
            end
            __stack_temporal_downsample_scale!(ds, factor, i, colrange, rowrange)
        end
    end
    ds
end

function stack_temporal_downsample!(ds, stack, factor; kwargs...)
    nr_ds, nc_ds, nff = size(ds)
    nr, nc, nf = size(stack)
    size_ok = nr_ds == nr && nc_ds == nc && nff * factor >= nf
    if ! size_ok
        throw(ArgumentError("Sizes mismatched"))
    end
    _stack_temporal_downsample!(ds, stack, factor; kwargs...)
end

function stack_temporal_downsample(::Type{T}, stack, factor;
                                   nt = nthreads()) where T
    nr, nc, nf = size(stack)
    nff = fld(nf, factor)
    ds = zeros(T, nr, nc, nff)
    _stack_temporal_downsample!(ds, stack, factor; nt)
end

stack_temporal_downsample(stack, factor; kwargs...) =
    stack_temporal_downsample(Float32, stack, factor; kwargs...)

function split_range(splitno, r, nsplit)
    nel = length(r)
    len, rem = divrem(nel, nsplit)
    if len == 0
        if splitno > rem
            rem = 0
        else
            len, rem = 1, 0
        end
    end
    f = first(r) + ((splitno-1) * len)
    l = f + len - 1
    if rem > 0
        if splitno <= rem
            f = f + (splitno - 1)
            l = l + splitno
        else
            f = f + rem
            l = l + rem
        end
    end
    return f:l
end

splits(r, nsplit) = map(x -> split_range(x, r, nsplit), 1:nsplit)

function _initialize_median_filters!(filterbank, inb, xr, yr, fr)
    for y in yr
        @inbounds for x in xr
            initialize_filter!(filterbank[x, y], view(inb, x, y, fr))
        end
    end
end

function _get_medians!(b, filterbank, xr, yr)
    for y in yr
        @inbounds for x in xr
            b[x, y] = get_median_value(filterbank[x, y])
        end
    end
end

function _update_medians!(filterbank, enteringb, exitingb, xr, yr)
    for y in yr
        @inbounds for x in xr
            update_filter!(filterbank[x, y], enteringb[x, y], exitingb[x, y])
        end
    end
end

function _update_subtract_medians!(outb, filterbank, refb, enteringb, exitingb,
                                   xr, yr)
    for y in yr
        @inbounds for x in xr
            update_filter!(filterbank[x, y], enteringb[x, y], exitingb[x, y])
            outb[x, y] = refb[x, y] - get_median_value(filterbank[x, y])
        end
    end
end

function median_filter_frames!(outb,
                               filterbank::AbstractMatrix{<:MedianFilter},
                               inb; nt = nthreads())
    insz = size(inb)
    insz == size(outb) || throw(ArgumentError("data sizes do not match"))
    nx, ny, nf = size(inb)
    size(filterbank) == (nx, ny) || throw(ArugmentError("filter size does not match"))
    nx == 0 || ny == 0 && return outb
    xr = 1:nx
    half_win = div(first(filterbank).count - 1, 2)
    init_range = centered_range(1, half_win, nf)
    medbuff = Matrix{Float64}(undef, nx, ny)
    if nt > 1
        yranges = splits(1:ny, nt)
        tasks = Vector{Task}(undef, nt)
        @inbounds for tno in 1:nt
            tasks[tno] = @spawn(
                _initialize_median_filters!(filterbank, inb, xr, yranges[tno],
                                           init_range)
            )
        end
        foreach(wait, tasks)
        @inbounds for tno in 1:nt
            tasks[tno] = @spawn _get_medians!(medbuff, filterbank, xr, yranges[tno])
        end
        foreach(wait, tasks)
        for i in 1 : half_win + 1
            @inbounds for tno in 1:nt
                tasks[tno] = @spawn(
                    _subtract_frame!(view(outb, :, :, i), view(inb, :, :, i),
                                     medbuff, xr, first(yranges[tno]),
                                     last(yranges[tno]))
                )
            end
            foreach(wait, tasks)
        end
        for i in half_win + 2 : nf - half_win - 1
            @inbounds for tno in 1:nt
                tasks[tno] = @spawn(
                    _update_subtract_medians!(view(outb, :, :, i), filterbank,
                                              view(inb, :, :, i),
                                              view(inb, :, :, i + half_win),
                                              view(inb, :, :, i - half_win - 1),
                                              xr, yranges[tno])
                )
            end
            foreach(wait, tasks)
        end
        @inbounds for tno in 1:nt
            tasks[tno] = @spawn(
                _update_medians!(filterbank, view(inb, :, :, nf),
                                 view(inb, :, :, nf - 2 * half_win - 1), xr,
                                 yranges[tno])
            )
        end
        foreach(wait, tasks)
        @inbounds for tno in 1:nt
            tasks[tno] = @spawn _get_medians!(medbuff, filterbank, xr, yranges[tno])
        end
        for i in nf - half_win : nf
            @inbounds for tno in 1:nt
                tasks[tno] = @spawn(
                    _subtract_frame!(view(outb, :, :, i), view(inb, :, :, i),
                                     medbuff, xr, first(yranges[tno]),
                                     last(yranges[tno]))
                )
            end
            foreach(wait, tasks)
        end
    else
        yrange = 1:ny
        _initialize_median_filters!(filterbank, inb, xr, yrange, init_range)
        _get_medians!(medbuff, filterbank, xr, yrange)
        @inbounds for i in 1 : half_win + 1
            _subtract_frame!(view(outb, :, :, i), view(inb, :, :, i), medbuff,
                             xr, 1, ny)
        end
        @inbounds for i in half_win + 2 : nf - half_win - 1
            _update_subtract_medians!(view(outb, :, :, i), filterbank,
                                      view(inb, :, :, i),
                                      view(inb, :, :, i + half_win),
                                      view(inb, :, :, i - half_win - 1), xr,
                                      yrange)
        end
        _update_medians!(filterbank, view(inb, :, :, nf),
                         view(inb, :, :, nf - 2 * half_win - 1), xr, yrange)
        _get_medians!(medbuff, filterbank, xr, yrange)
        for i in nf - half_win : nf
            _subtract_frame!(view(outb, :, :, i), view(inb, :, :, i), medbuff,
                             xr, 1, ny)
        end
    end
    outb
end

function median_filter_frames!(outb, inb::AbstractArray, filter_npt,
                               valrange::AbstractRange{T}; discrete = false,
                               kwargs...) where T<:Number
    nx, ny, nf = size(inb)
    half_win = div(filter_npt - 1, 2)
    nwin = 2 * half_win + 1
    if discrete
        filterbank = Matrix{MedianFilter{T, true}}(undef, nx, ny)
    else
        filterbank = Matrix{MedianFilter{T, false}}(undef, nx, ny)
    end
    for i in eachindex(filterbank)
        filterbank[i] = MedianFilter(valrange, nwin, discrete)
    end
    median_filter_frames!(outb, filterbank, inb; kwargs...)
end

end # module
