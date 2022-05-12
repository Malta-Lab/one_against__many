# multitask_code
* [ ] Juntar os dois repositórios
  * [ ] Unificar classes semelhantes
  * [ ] Unificar modo de definir argumentos
* [ ] Subir na nuvem os melhores modelos, que vão servir como baseline


## code_search

### Download dataset:
* https://brpucrs-my.sharepoint.com/:u:/g/personal/otavio_parraga_edu_pucrs_br/ETD3f0jARYBHqUlQREqN750BL0BojS2mRg3IHF6KkSycDw?e=LOzaM2
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

### Baselines:
* For CodeT5 and CodeBERT = base with prefix
* For GraphCodeBERT = base without prefix