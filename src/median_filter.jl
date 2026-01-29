mutable struct MedianFilter{T,D}
    cumulative::Vector{Int}
    count::Int
    last_median_pos::Int
    first_bin::T
    bin_width::T
end

function MedianFilter(r::AbstractRange{T}, count::Integer, discrete = false) where {T}
    l = length(r) - 1
    cum = Vector{Int}(undef, l)
    first_bin = first(r)
    bin_width = step(r)
    MedianFilter{T,discrete}(cum, count, 1, first_bin, bin_width)
end

bin_ndx(m::MedianFilter, obs) =
    clamp(convert(Int, fld(obs - m.first_bin, m.bin_width)) + 1, 1, length(m.cumulative))


function initialize_filter!(m::MedianFilter, vals::AbstractVector)
    length(vals) != m.count && throw(ArgumentError("Wrong number of vals"))
    fill!(m.cumulative, 0)
    @inbounds for v in vals
        i = bin_ndx(m, v)
        m.cumulative[i] += 1
    end
    last_median_pos = 0
    half_point = div(m.count + 1, 2)
    for i = 1:(length(m.cumulative)-1)
        m.cumulative[i+1] += m.cumulative[i]
        last_median_pos = ifelse(
            (m.cumulative[i] < half_point) & (m.cumulative[i+1] >= half_point),
            i + 1,
            last_median_pos,
        )
    end
    m.last_median_pos = last_median_pos
    m
end

function get_median_value(m::MedianFilter{<:Any,false})
    last_median_pos = m.last_median_pos
    nbin = length(m.cumulative)
    lower_bnd = m.bin_width * (last_median_pos - 1) + m.first_bin
    if last_median_pos > 1
        @inbounds cum_lower = m.cumulative[last_median_pos-1]
    else
        cum_lower = 0
    end
    @inbounds bin_freq = m.cumulative[last_median_pos] - cum_lower
    lower_bnd + m.bin_width * (m.count / 2 - cum_lower) / bin_freq
end

function get_median_value(m::MedianFilter{<:Any,true})
    lastpos = m.last_median_pos
    nbin = length(m.cumulative)
    upper_bin = convert(Float64, m.bin_width * (lastpos - 1) + m.first_bin)
    lower_half_point = div(m.count, 2)
    if isodd(nbin) || lastpos == 1 || m.cumulative[lastpos-1] < lower_half_point
        m = upper_bin
    else
        lower_bin = m.bin_width * (lastpos - 2) + m.first_bin
        m = (upper_bin + lower_bin) / 2
    end
    m
end

function update_filter!(m::MedianFilter, entering, exiting)
    nbins = length(m.cumulative)
    nbins > 1 || return
    bin_enter = bin_ndx(m, entering)
    bin_exit = bin_ndx(m, exiting)
    start_bin = min(bin_enter, bin_exit)
    stop_bin = max(bin_enter, bin_exit)
    adj_bin_range = start_bin:(stop_bin-1)
    d = ifelse(bin_enter <= bin_exit, 1, -1)
    @inbounds @simd for binno in adj_bin_range
        m.cumulative[binno] += d
    end
    lastpos = m.last_median_pos
    if start_bin <= lastpos <= stop_bin
        half_pos = div(m.count + 1, 2)
        @inbounds goright = m.cumulative[lastpos] < half_pos
        if goright
            @inbounds for i = (lastpos+1):nbins
                if m.cumulative[i] >= half_pos
                    m.last_median_pos = i
                    break
                end
            end
        else
            set_pos = false
            @inbounds for i = (lastpos-1):-1:1
                if m.cumulative[i] < half_pos
                    m.last_median_pos = i + 1
                    set_pos = true
                    break
                end
            end
            if ! set_pos
                m.last_median_pos = 1
            end
        end
    end
    nothing
end

function centered_range(c::Signed, half_win::Signed, m::Signed)
    b, e = clip_interval_duration(c - half_win, c + half_win, 1, m)
    b:e
end

function calc_median_filter!(out, m::MedianFilter, x)
    nx = length(x)
    half_win = div(m.count - 1, 2)
    r = centered_range(1, half_win, nx)
    initialize_filter!(m, view(x, r))
    @inbounds out[1:(half_win+1)] .= get_median_value(m)
    @inbounds for outpos = (half_win+2):(nx-half_win-1)
        update_filter!(m, x[outpos+half_win], x[outpos-half_win-1])
        out[outpos] = get_median_value(m)
    end
    update_filter!(m, x[end], x[end-2*half_win-1])
    @inbounds out[(nx-half_win):end] .= get_median_value(m)
    out
end

function calc_median_filter!(out, x, valrange, n; discrete = false)
    nx = length(x)
    half_win = div(n - 1, 2)
    nwin = 2 * half_win + 1
    m = MedianFilter(valrange, nwin, discrete)
    calc_median_filter!(out, m, x)
end
