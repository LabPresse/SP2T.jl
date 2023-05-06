function pixel2frame!(frame::AbstractMatrix{Bool}, pixels::AbstractVector{<:Integer})
    frame[pixels.+1] .= true
end

function readSPADbinary(path::String, sizex::Int, sizey::Int)

    pixelwithsignal = open(path) do file
        reinterpret(UInt32, read(file))
    end

    pixelnum = sizex * sizey

    delimiters = findall(pixelwithsignal .== pixelnum)
    framenum = length(delimiters)

    data = zeros(Bool, sizey, sizex, count(pixelwithsignal .== sizex * sizey))

    start = 1
    for i = 1:framenum
        pixel2frame!(view(data, :, :, i), @view pixelwithsignal[start:delimiters[i]-1])
        start = delimiters[i] + 1
    end
    return data
end

"/home/lancexwq/Dropbox (ASU)/Weiqing-Nathan/SPAD512 Beads/Orange beads 1-bit images/RAW0003-0000.bin"