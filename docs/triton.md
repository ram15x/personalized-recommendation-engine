# Experimental Triton ranker interface

`triton/model_repository/dcn_ranker/` contains a Triton Python backend model configuration and implementation for the saved experimental DCN ranker. The model expects aligned, nonempty vectors `USER_ID` (INT64), `ITEM_ID` (INT64), and `ITEMKNN_SCORE` (FP32), with one user per request. It returns `RANK_SCORE` (FP32) and `SUPPORTED` (BOOL). A false `SUPPORTED` entry means the item is outside the saved vocabulary; its score placeholder is zero and must not be used for ordering. Unknown users and malformed inputs return errors. The caller is responsible for candidate generation and fallback ordering.

The backend expects the project code on `PYTHONPATH` and `NEXVIA_EXPERIMENTAL_RANKER_DIR` pointing to the saved model and mapping. To avoid installing TensorFlow in Triton, export the saved Keras DCN weights once on a host with the project's TensorFlow environment:

```powershell
python -m scripts.triton_numpy_ranker reports/development/dcn-itemknn-candidate-ranker-50-60-to-60-70-v1
```

This creates `triton_numpy_weights.npz` and a `triton_numpy_weights.npz.sha256` sidecar, without changing the base artifact manifest. The NumPy inference code checks the saved artifact hashes and support metadata. It uses the same DCN embedding, cross, dense, and output weights; dropout is inactive at inference. Triton does not supply the ItemKNN features itself.

The live-tested command used the full CPU Triton image. Mount the project read-only, including the generated local artifact, and the model repository:

```powershell
$project = (Get-Location).Path
docker run --rm -p 127.0.0.1:8000:8000 `
  -e PYTHONPATH=/workspace/project `
  -e NEXVIA_EXPERIMENTAL_RANKER_DIR=/workspace/project/reports/development/dcn-itemknn-candidate-ranker-50-60-to-60-70-v1 `
  --mount "type=bind,source=$project,target=/workspace/project,readonly" `
  --mount "type=bind,source=$project\triton\model_repository,target=/models,readonly" `
  nvcr.io/nvidia/tritonserver:25.10-py3 `
  tritonserver --model-repository=/models --exit-on-error=true
```

The tested server was Triton 2.62.0 on CPU. `/v2/health/ready` and `/v2/models/dcn_ranker/ready` returned HTTP 200. One live inference for user 139 over items `[1, 2, 3, 5, 6]` returned rank scores `[3.1989660, 2.9194598, 2.7488132, 3.0142066, 3.8532543]`; all support flags were true. NumPy and TensorFlow outputs differed by at most `2.3841858e-7` over the 32-pair parity check and the live request. This confirms inference compatibility, not a quality gain. The candidate did not beat ItemKNN in its offline development comparison and is not the default. The full Triton image occupied 19,885,930,811 bytes locally and was removed after the check; repulling it is required to run this again.

Triton server load and network inference are verified as described above. The ranker remains experimental and did not beat ItemKNN in the saved development comparison. See [NVIDIA's Python backend documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html) and [model configuration guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html).
