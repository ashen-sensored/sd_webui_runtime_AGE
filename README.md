# sd_webui_runtime_ensembling
!!! This extension is not yet compatible with sd-webui-runtime-block-merge or sd-webui-layer-controller, please disable them before enabling this extension.

Stable Diffusion runtime ensembling POC extension for Webui. Intial idea inspired by https://deepimagination.cc/eDiff-I/.

Please be advised that by default this extension has high VRAM requirement (up to 3X original usage during inference).
It is possible to mitigate the VRAM requirement by adding --lowvram --no-half to your webui COMMANDLINE_ARGS, at the cost of massively decreaced inference performance and increased RAM usage.

Example:

Model A: Pastel Mix https://huggingface.co/andite/pastel-mix
![Screenshot from 2023-01-28 02-08-12](https://user-images.githubusercontent.com/121544382/215283961-aac4a741-05c0-489f-80fb-f90df8f47586.png)

Runtime Ensembling With Model B (Ours)
![Screenshot from 2023-01-28 07-06-51](https://user-images.githubusercontent.com/121544382/215284460-0edb5b7f-19ea-4b44-af73-57f7af652ac5.png)

0.5 Weighted Merge With Model B (Theirs)
![Screenshot from 2023-01-28 10-25-16](https://user-images.githubusercontent.com/121544382/215284441-7820a7d3-dc06-4aac-a3c0-e7afe8695561.png)

Model B:
![Screenshot from 2023-01-28 07-29-02](https://user-images.githubusercontent.com/121544382/215284486-e4a7ed74-a03f-4cec-a421-66f5995d28b7.png)
