"""Facts about benchmark detectors that change how they are ranked."""

# detectors fed the expert hypnogram (reproduction runs): never ranked against stage-free detectors
USES_EXPERT_STAGING = {"sumov2_n2", "moelle2011_faithful_nrem", "a7_n2base", "a7_incontext_n2base"}
# detectors whose score on a dataset is in-sample (trained or built on that dataset's labels):
# reported as a reference tier, ranked separately from out-of-sample detectors
IN_SAMPLE = {"dreams_spindles": {"dreams_auto", "openspindlenet"}, "dreams_kc": {"dreams_auto"}}
# near-duplicate configurations of one algorithm (for family-collapsed rank statistics)
FAMILY = {"a7_incontext": "a7", "moelle2011_faithful": "wonambi_moelle2011", "wonambi_lacourse2018": "a7",
          "sumo": "sumov2", "sssm_v2": "sssm", "sssm_strict": "sssm",
          "yasa_loose": "yasa_default", "yasa_dreams_tuned": "yasa_default", "wonambi_fasst2": "wonambi_fasst"}
# left out of the NAP point-scoring ranking: expert-staging reproduction runs and K-complex detectors
NAP_EXCLUDE = USES_EXPERT_STAGING | {"spinky_kc", "yasa_sw_kc"}
