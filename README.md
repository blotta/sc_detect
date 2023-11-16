1. `pip install -r requirements.txt`

2. Colocar todas as imagens de [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) na pasta `data/all_images` e `HAM10000_metadata.csv` na como `data/HAM10000_metadata.csv`

3. Rodar `reorganize_data.py`, que separa imagens em pastas em `data/reorganized/`

4. Rodar `train.py` para criar o modelo.

5. Testar uma imagem em `data/test_images` com `model_inputtest.py`. Substitua o arquivo de `model` pelo gerado no passo anterior.