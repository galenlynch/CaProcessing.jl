module CaProcessing

import Base.Threads.@spawn

# Stdlib
using Mmap, Statistics
# Public packages
using FixedPointNumbers, ImageCore
# Private packages
using GLUtilities: indices_above_thresh

export demin,
    demin!,
    avg_intensities,
    max_intensities,
    clip_segments_thr,
    map_to_8bit

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

function clip_segments_thr(imgs, thr; nt = Threads.nthreads(), x = :, y = :)
    nf = size(imgs, 3)
    clipped_imgs = view(imgs, x, y, 1:nf)
    intensities = max_intensities(imgs, nt = nt)
    open_pers = indices_above_thresh(intensities, thr)
    nseg = size(open_pers, 2)

    segments = Vector{typeof(clipped_imgs)}(undef, nseg)
    for segno in 1:nseg
        segments[segno] = view(imgs, x, y, open_pers[1, segno]:open_pers[2, segno])
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

end # module
