"""Contract checks with a backend stub; these do not run Triton itself."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_triton_backend_handles_known_unknown_and_bad_shapes(monkeypatch):
    class Tensor:
        def __init__(self, name, values):
            self.name, self.values = name, values
        def as_numpy(self):
            return self.values

    class Response:
        def __init__(self, output_tensors=None, error=None):
            self.output_tensors, self.error = output_tensors, error

    stub = SimpleNamespace(Tensor=Tensor, InferenceResponse=Response,
                           TritonError=lambda value: value,
                           get_input_tensor_by_name=lambda request, name: request.get(name))
    monkeypatch.setitem(sys.modules, "triton_python_backend_utils", stub)
    path = Path(__file__).parent / "triton/model_repository/dcn_ranker/1/model.py"
    spec = importlib.util.spec_from_file_location("triton_test_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Ranker:
        def score(self, user_id, items, scores):
            assert user_id == 7 and list(items) == [10, 99]
            return [2.5, None]

    model = module.TritonPythonModel()
    model.ranker = Ranker()
    def request(users):
        return {"USER_ID": Tensor("USER_ID", np.array(users, dtype=np.int64)),
                "ITEM_ID": Tensor("ITEM_ID", np.array([10, 99], dtype=np.int64)),
                "ITEMKNN_SCORE": Tensor("ITEMKNN_SCORE", np.array([.8, .1], dtype=np.float32))}
    good, bad = model.execute([request([7, 7]), request([7, 8])])
    assert good.error is None
    assert good.output_tensors[0].values.tolist() == [2.5, 0.0]
    assert good.output_tensors[1].values.tolist() == [True, False]
    assert "one user" in bad.error
