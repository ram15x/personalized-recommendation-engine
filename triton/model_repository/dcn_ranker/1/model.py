"""Triton Python backend for the experimental saved candidate ranker."""
import os
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # The project must be on PYTHONPATH and its saved artifact mounted.
        from scripts.triton_numpy_ranker import NumpyCandidateRanker

        directory = os.environ.get("NEXVIA_EXPERIMENTAL_RANKER_DIR")
        if not directory:
            raise RuntimeError("NEXVIA_EXPERIMENTAL_RANKER_DIR must identify the saved ranker directory")
        self.ranker = NumpyCandidateRanker(Path(directory))

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                users = pb_utils.get_input_tensor_by_name(request, "USER_ID").as_numpy()
                items = pb_utils.get_input_tensor_by_name(request, "ITEM_ID").as_numpy()
                retrieval = pb_utils.get_input_tensor_by_name(request, "ITEMKNN_SCORE").as_numpy()
                if (users.ndim != 1 or items.ndim != 1 or retrieval.ndim != 1
                        or not len(users) or users.shape != items.shape or users.shape != retrieval.shape
                        or np.any(users != users[0])):
                    raise ValueError("Inputs must be nonempty aligned vectors for one user.")
                scores = self.ranker.score(int(users[0]), items, retrieval)
                supported = np.asarray([score is not None for score in scores], dtype=np.bool_)
                values = np.asarray([0.0 if score is None else score for score in scores], dtype=np.float32)
                responses.append(pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("RANK_SCORE", values),
                    pb_utils.Tensor("SUPPORTED", supported),
                ]))
            except (ValueError, TypeError, AttributeError) as exc:
                responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc))))
        return responses
