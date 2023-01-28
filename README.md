# sd_webui_runtime_ensembling
Stable Diffusion runtime ensembling POC extension for Webui. Intial idea inspired by https://deepimagination.cc/eDiff-I/.

Please be advised that by default this extension has high VRAM requirement (up to 3X original usage during inference).
It is possible to mitigate the VRAM requirement by adding --lowvram --no-half to your webui COMMANDLINE_ARGS, at the cost of massively decreaced inference performance and increased RAM usage.

Example:

Model A: Pastel Mix https://huggingface.co/andite/pastel-mix
![Screenshot from 2023-01-28 02-08-12](https://user-images.githubusercontent.com/121544382/215283961-aac4a741-05c0-489f-80fb-f90df8f47586.png)

0.5 Weighted Merge With Model B (Theirs)
![Screenshot from 2023-01-28 02-05-45](https://user-images.githubusercontent.com/121544382/215284116-1eb53920-3409-4064-965b-6175b431de8d.png)

Runtime Ensembling With Model B (Ours)
![Screenshot from 2023-01-28 02-09-49](https://user-images.githubusercontent.com/121544382/215284145-5d33327e-2cfa-48c2-b039-5b4c0bae8881.png)
