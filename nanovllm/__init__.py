from nanovllm.sampling_params import SamplingParams

def __getattr__(name):
    # Lazy-import LLM so protocol tests do not pull torch/transformers.
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
