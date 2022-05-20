# multitask_code
* [ ] Subir na nuvem os melhores modelos, que vão servir como baseline


## code_search

### Download dataset:
* Downloaded from [CodeBERT](https://github.com/microsoft/CodeBERT)
* [OneDrive](https://brpucrs-my.sharepoint.com/:u:/g/personal/otavio_parraga_edu_pucrs_br/ETD3f0jARYBHqUlQREqN750BL0BojS2mRg3IHF6KkSycDw?e=LOzaM2)
* To clean Python split: `python preprocess.py --mode clean`

### Baselines:
* go: StepLR
* java: StepLR
* javascript: Linear
* php: StepLR
* python: StepLR
* ruby: StepLR

## Code2UnitTest

### Dataset:
* Downloaded from Microsoft [Methods2Test](https://github.com/microsoft/methods2test)
  * To extract files: `python preprocess.py --mode extract`
* [OneDrive](https://brpucrs-my.sharepoint.com/:f:/g/personal/otavio_parraga_edu_pucrs_br/EkBIRbGh4xNDm6dXfvUuizsBlCT1Gw3A_pUsVuI2u4QfSg?e=FLFLBq)

### Baselines:
* For CodeT5 and CodeBERT = base with prefix
* For GraphCodeBERT = base without prefix

## For MultiTask Learning:
* [Datasets](https://brpucrs-my.sharepoint.com/:f:/g/personal/otavio_parraga_edu_pucrs_br/EmHyWGgXsdRJrMGOTRPiqwUBWVBOZOKtdwu0J-j9LcwRcQ?e=ENaOih)