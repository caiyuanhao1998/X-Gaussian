# X-Gaussian CT CUDA backend

Build with `bash install_ct.sh` from the repository root. This separately named
package supplies additive Gaussian projection and voxel-query autograd operators.
It does not replace the original X-Gaussian renderer.

Adapted from R2-Gaussian; see [UPSTREAM.md](UPSTREAM.md) and [LICENSE.md](LICENSE.md).
Vendored GLM headers retain their original license in `third_party/glm/copying.txt`.
