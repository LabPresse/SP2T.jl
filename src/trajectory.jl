update_𝕩!(s::FullSample, prior::Sampleable, param::ExperimentalParameter) =
    simulate!(view_𝕩(s), prior, s.D, param.period)