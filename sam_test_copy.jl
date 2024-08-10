using CUDA
using QuasiGrad
using BenchmarkTools

function GPU_calc()
    # define test
    InFile                = "./scenario_001.json"
    NewTimeLimitInSeconds = 600.0
    Division              = 1
    NetworkModel          = "test"
    AllowSwitching        = 0

    jsn = QuasiGrad.load_json(InFile)
    adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
        QuasiGrad.base_initialization(jsn, Div=Division, hpc_params=true, line_switching=AllowSwitching);

    # ac line flows
    # function acline_flows!(grd::QuasiGrad.Grad, idx::QuasiGrad.Index, prm::QuasiGrad.Param, qG::QuasiGrad.QG, stt::QuasiGrad.State, sys::QuasiGrad.System)
    # line parameters
    g_sr = CuArray(prm.acline.g_sr)
    b_sr = CuArray(prm.acline.b_sr)
    b_ch = CuArray(prm.acline.b_ch)
    g_fr = CuArray(prm.acline.g_fr)
    b_fr = CuArray(prm.acline.b_fr)
    g_to = CuArray(prm.acline.g_to)
    b_to = CuArray(prm.acline.b_to)

    # call penalty costs
    cs = prm.vio.s_flow * qG.scale_c_sflow_testing

    # Organize the relevant line values for each time step and transfer to GPU
    # Flatten the arrays and use views to handle each time step
    vm_fr = CuArray(vcat([stt.vm[t][idx.acline_fr_bus] for t in prm.ts.time_keys]...))
    va_fr = CuArray(vcat([stt.va[t][idx.acline_fr_bus] for t in prm.ts.time_keys]...))
    vm_to = CuArray(vcat([stt.vm[t][idx.acline_to_bus] for t in prm.ts.time_keys]...))
    va_to = CuArray(vcat([stt.va[t][idx.acline_to_bus] for t in prm.ts.time_keys]...))

    # Create output CuArrays
    cos_ftp = similar(vm_fr)
    sin_ftp = similar(va_fr)
    vff = similar(vm_fr)
    vtt = similar(vm_to)
    vft = similar(vm_fr)

    pfr = similar(vm_fr)
    qfr = similar(vm_fr)
    pto = similar(vm_fr)
    qto = similar(vm_fr)
    acline_sfr = similar(vm_fr)
    acline_sto = similar(vm_fr)


    # Perform computations on GPU
    for tidx in 1:length(prm.ts.time_keys)
        tii = prm.ts.time_keys[tidx]

        # duration
        dt = prm.ts.duration[tii]

        # Access slices of the flattened arrays corresponding to this time step
        offset = (tidx - 1) * length(stt.vm_fr[tii])
        range = offset + 1 : offset + length(stt.vm_fr[tii])

        # Compute line values on the GPU
        @views begin
            cos_ftp[range] .= cos.(va_fr[range] .- va_to[range])
            sin_ftp[range] .= sin.(va_fr[range] .- va_to[range])
            vff[range] .= vm_fr[range] .^ 2
            vtt[range] .= vm_to[range] .^ 2
            vft[range] .= vm_fr[range] .* vm_to[range]

            # Evaluate the function for active and reactive power flow
            pfr[range] .= (g_sr .+ g_fr) .* vff[range] .+ (-g_sr .* cos_ftp[range] .- b_sr .* sin_ftp[range]) .* vft[range]
            qfr[range] .= (-b_sr .- b_fr .- b_ch ./ 2.0) .* vff[range] .+ (b_sr .* cos_ftp[range] .- g_sr .* sin_ftp[range]) .* vft[range]
            acline_sfr[range] .= sqrt.(pfr[range].^2 .+ qfr[range].^2)

            pto[range] .= (g_sr .+ g_to) .* vtt[range] .+ (-g_sr .* cos_ftp[range] .+ b_sr .* sin_ftp[range]) .* vft[range]
            qto[range] .= (-b_sr .- b_to .- b_ch ./ 2.0) .* vtt[range] .+ (b_sr .* cos_ftp[range] .+ g_sr .* sin_ftp[range]) .* vft[range]
            acline_sto[range] .= sqrt.(pto[range].^2 .+ qto[range].^2)

        end
    end
end
@btime GPU_calc()