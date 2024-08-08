

# define test
InFile                = "./data/C3E3.1_20230629/D1/C3E3N00617D1/scenario_001.json"
NewTimeLimitInSeconds = 600.0
Division              = 1
NetworkModel          = "test"
AllowSwitching        = 0

jsn = QuasiGrad.load_json(InFile1)
adm, cgd, ctg, flw, grd, idx, lbf, mgd, ntk, prm, qG, scr, stt, sys, upd = 
    QuasiGrad.base_initialization(jsn, Div=Division, hpc_params=true, line_switching=AllowSwitching);

# ac line flows
# function acline_flows!(grd::QuasiGrad.Grad, idx::QuasiGrad.Index, prm::QuasiGrad.Param, qG::QuasiGrad.QG, stt::QuasiGrad.State, sys::QuasiGrad.System)
# line parameters
g_sr = prm.acline.g_sr
b_sr = prm.acline.b_sr
b_ch = prm.acline.b_ch
g_fr = prm.acline.g_fr
b_fr = prm.acline.b_fr
g_to = prm.acline.g_to
b_to = prm.acline.b_to

pfr = CuArray(copy(stt.pfr))

# call penalty costs
cs = prm.vio.s_flow * qG.scale_c_sflow_testing

# loop over time -- use per=core
Threads.@threads for tii in prm.ts.time_keys
# => @floop ThreadedEx(basesize = qG.nT ÷ qG.num_threads) for tii in prm.ts.time_keys

    # duration
    dt = prm.ts.duration[tii]

    # organize relevant line values
    stt.vm_fr[tii] .= @view stt.vm[tii][idx.acline_fr_bus]
    stt.va_fr[tii] .= @view stt.va[tii][idx.acline_fr_bus]
    stt.vm_to[tii] .= @view stt.vm[tii][idx.acline_to_bus]
    stt.va_to[tii] .= @view stt.va[tii][idx.acline_to_bus]
    
    # tools
    @turbo stt.cos_ftp[tii] .= QuasiGrad.LoopVectorization.cos_fast.(stt.va_fr[tii] .- stt.va_to[tii])
    @turbo stt.sin_ftp[tii] .= QuasiGrad.LoopVectorization.sin_fast.(stt.va_fr[tii] .- stt.va_to[tii])
    @turbo stt.vff[tii]     .= QuasiGrad.LoopVectorization.pow_fast.(stt.vm_fr[tii],2)
    @turbo stt.vtt[tii]     .= QuasiGrad.LoopVectorization.pow_fast.(stt.vm_to[tii],2)
    @turbo stt.vft[tii]     .= stt.vm_fr[tii].*stt.vm_to[tii]
    
    # evaluate the function? we always need to in order to get the grd
    #
    # active power flow -- from -> to
    @turbo stt.pfr[tii] .= (g_sr.+g_fr).*stt.vff[tii] .+ (.-g_sr.*stt.cos_ftp[tii] .- b_sr.*stt.sin_ftp[tii]).*stt.vft[tii]
    @turbo stt.acline_pfr[tii] .= stt.u_on_acline[tii].*stt.pfr[tii]
    
    # reactive power flow -- from -> to
    @turbo stt.qfr[tii] .= (.-b_sr.-b_fr.-b_ch./2.0).*stt.vff[tii] .+ (b_sr.*stt.cos_ftp[tii] .- g_sr.*stt.sin_ftp[tii]).*stt.vft[tii]
    @turbo stt.acline_qfr[tii] .= stt.u_on_acline[tii].*stt.qfr[tii]
    
    # apparent power flow -- to -> from
    @turbo stt.acline_sfr[tii] .= QuasiGrad.LoopVectorization.sqrt_fast.(QuasiGrad.LoopVectorization.pow_fast.(stt.acline_pfr[tii],2) .+ QuasiGrad.LoopVectorization.pow_fast.(stt.acline_qfr[tii],2))
    
    # active power flow -- to -> from
    @turbo stt.pto[tii] .= (g_sr.+g_to).*stt.vtt[tii] .+ (.-g_sr.*stt.cos_ftp[tii] .+ b_sr.*stt.sin_ftp[tii]).*stt.vft[tii]
    @turbo stt.acline_pto[tii] .= stt.u_on_acline[tii].*stt.pto[tii]
    
    # reactive power flow -- to -> from
    @turbo stt.qto[tii] .= (.-b_sr.-b_to.-b_ch./2.0).*stt.vtt[tii] .+ (b_sr.*stt.cos_ftp[tii] .+ g_sr.*stt.sin_ftp[tii]).*stt.vft[tii]
    @turbo stt.acline_qto[tii] .= stt.u_on_acline[tii].*stt.qto[tii]

    # apparent power flow -- to -> from
    @turbo stt.acline_sto[tii] .= QuasiGrad.LoopVectorization.sqrt_fast.(QuasiGrad.LoopVectorization.pow_fast.(stt.acline_pto[tii],2) .+ QuasiGrad.LoopVectorization.pow_fast.(stt.acline_qto[tii],2))
end

# 0. understand this code and its relation to (148)-(151): https://gocompetition.energy.gov/sites/default/files/Challenge3_Problem_Formulation_20230515.pdf
# 1. get above to kind of run (get rid of all the turbo stuff and loop vectorization)
# 2. how can you run this on GPUs? use CuArrays, probably