#!/usr/bin/env python3

# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import base64
import os
import sys
from io import BytesIO
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

_src_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


class _MockSamplingParams:
    __annotations__ = {}

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


_pb_utils = ModuleType("triton_python_backend_utils")
_pb_utils.get_input_tensor_by_name = MagicMock()
_pb_utils.Tensor = MagicMock()
_pb_utils.InferenceResponse = MagicMock()
sys.modules["triton_python_backend_utils"] = _pb_utils

_vllm_modules = [
    "vllm",
    "vllm.inputs",
    "vllm.inputs.data",
    "vllm.lora",
    "vllm.lora.request",
    "vllm.outputs",
    "vllm.pooling_params",
    "vllm.utils",
    "vllm.sampling_params",
    "vllm.engine",
    "vllm.engine.arg_utils",
    "vllm.engine.protocol",
    "vllm.usage",
    "vllm.usage.usage_lib",
    "vllm.v1",
    "vllm.v1.metrics",
    "vllm.v1.metrics.loggers",
]
for _mod_name in _vllm_modules:
    sys.modules[_mod_name] = ModuleType(_mod_name)

sys.modules["vllm.utils"].random_uuid = lambda: "test-uuid"
sys.modules["vllm.outputs"].RequestOutput = MagicMock
sys.modules["vllm.outputs"].EmbeddingOutput = MagicMock
sys.modules["vllm.outputs"].EmbeddingRequestOutput = MagicMock


class _SubscriptableMock:
    def __class_getitem__(cls, item):
        return cls


sys.modules["vllm.outputs"].PoolingRequestOutput = _SubscriptableMock
sys.modules["vllm.pooling_params"].PoolingParams = MagicMock
sys.modules["vllm.lora.request"].LoRARequest = MagicMock
sys.modules["vllm.inputs.data"].TokensPrompt = MagicMock
sys.modules["vllm.sampling_params"].SamplingParams = _MockSamplingParams
sys.modules["vllm.sampling_params"].StructuredOutputsParams = MagicMock
sys.modules["vllm.engine.arg_utils"].AsyncEngineArgs = MagicMock
sys.modules["vllm.engine.protocol"].EngineClient = MagicMock
_usage_context_mock = MagicMock()
_usage_context_mock.OPENAI_API_SERVER = "openai_api_server"
sys.modules["vllm.usage.usage_lib"].UsageContext = _usage_context_mock
sys.modules["vllm.v1.metrics.loggers"].StatLoggerFactory = MagicMock

from utils.request import GenerateRequest  # noqa: E402


def _make_tensor(value):
    tensor = MagicMock()
    tensor.as_numpy.return_value = value
    return tensor


_ADDITIONAL_OUTPUT_KEYS = {
    "return_finish_reason": None,
    "return_cumulative_logprob": None,
    "return_logprobs": None,
    "return_num_input_tokens": None,
    "return_num_output_tokens": None,
}


def _build_generate_request(tensors_map):
    def get_tensor(req, name):
        return tensors_map.get(name)

    _pb_utils.get_input_tensor_by_name = MagicMock(side_effect=get_tensor)

    mock_request = MagicMock()
    mock_request.parameters.return_value = "{}"
    logger = MagicMock()

    return GenerateRequest(mock_request, MagicMock(), np.object_, logger)


def _base_tensors(**overrides):
    tensors = {
        "text_input": None,
        "image": None,
        "audio": None,
        "sample_rate": None,
        "stream": None,
        "exclude_input_in_output": None,
        "sampling_parameters": None,
        **_ADDITIONAL_OUTPUT_KEYS,
    }
    tensors.update(overrides)
    return tensors


def test_audio_input_with_string_prompt():
    audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    audio_tensor = _make_tensor(audio_data)

    tensors = _base_tensors(
        text_input=_make_tensor(np.array([b"Describe the audio"])),
        audio=audio_tensor,
    )

    req = _build_generate_request(tensors)
    prompt, stream, prepend_input, params, additional = req._get_input_tensors()

    assert isinstance(prompt, dict)
    assert prompt["prompt"] == "Describe the audio"
    assert "audio" in prompt["multi_modal_data"]
    audio_out, sr_out = prompt["multi_modal_data"]["audio"]
    np.testing.assert_array_equal(audio_out, audio_data)
    assert sr_out == 16000


def test_audio_input_with_custom_sample_rate():
    audio_data = np.array([0.5, -0.5], dtype=np.float32)
    audio_tensor = _make_tensor(audio_data)
    sr_tensor = _make_tensor(np.array([44100]))

    tensors = _base_tensors(
        text_input=_make_tensor(np.array([b"Transcribe"])),
        audio=audio_tensor,
        sample_rate=sr_tensor,
    )

    req = _build_generate_request(tensors)
    prompt, *_ = req._get_input_tensors()

    _, sr_out = prompt["multi_modal_data"]["audio"]
    assert sr_out == 44100


def test_audio_merges_with_existing_image_prompt():
    audio_data = np.array([1.0, 2.0], dtype=np.float32)
    audio_tensor = _make_tensor(audio_data)

    img = Image.new("RGB", (1, 1))
    buf = BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    image_tensor = _make_tensor(np.array([img_b64.encode("utf-8")]))

    tensors = _base_tensors(
        text_input=_make_tensor(np.array([b"Describe both"])),
        image=image_tensor,
        audio=audio_tensor,
    )

    req = _build_generate_request(tensors)
    prompt, *_ = req._get_input_tensors()

    assert isinstance(prompt, dict)
    assert "image" in prompt["multi_modal_data"]
    assert "audio" in prompt["multi_modal_data"]


def test_no_audio_input_leaves_prompt_unchanged():
    tensors = _base_tensors(
        text_input=_make_tensor(np.array([b"Just text"])),
    )

    req = _build_generate_request(tensors)
    prompt, *_ = req._get_input_tensors()

    assert isinstance(prompt, str)
    assert prompt == "Just text"
