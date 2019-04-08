# Projeto I: Caracterização e simulação de canais

Esse projeto divide-se em duas partes, aonde a primeira consiste da caracterização do canal através do qual sinais dados são transmitidos e a segunda consiste da simulação do canal móvel utilizando modelos clássicos.

A caracterização se dá pela decomposição dos efeitos de perda de potência em um sinal em relação à distância entre o transmissor e o receptor. Os dados analisados foram gerados sinteticamente e, portanto, a abordagem descrita é de engenharia reversa. Tenta-se, então, encontrar os parâmetros de descrição do canal ao separar os diferentes efeitos de perda sobre o sinal. A partir de então, utilizam-se os aprendizados adquiridos para analisar um sinal real fornecido.

A simulação, outra etapa crucial dos estudos de comunicações móveis, se dá pelo uso de dois modelos clássicos: Clarke/Gans e Jakes. Ambos com suas respectivas características e detalhes.

# Instalação

Crie um ambiente virtual para instalar as dependências localmente no projeto. Note que as dependências desse projeto foram gerenciadas utilizando o Poetry.

Sendo assim, é necessário instalá-lo, siga as instruções fornecidas no [projeto oficial](https://github.com/sdispater/poetry).
Fique a vontade para ler informações adicionais em sua [documentação](https://poetry.eustace.io/docs/basic-usage/).

Por favor note que esse projeto requer, no mínimo, Python 3.6 para que possa roda, uma vez que usa f-strings, [adicionadas ao Python nessa versão](https://www.python.org/dev/peps/pep-0498/).

Dado que você possui uma versão compatível do Python, você já deve possuir os módulos distutils, pip e venv instalados. Contudo, dependendo do seu sistema operacional e da respectiva distribuição, é possível que esses módulos tenham sido removidos da instalação padrão, sendo necessário instalá-los individualmente. No caso do Ubuntu, deve-se utilizar:
```shell
apt-get install python3-distutils python3-venv python3-pip
```

Uma vez que se tenha o Poetry devidamente instalado, crie um ambiente virtual do Python nessa pasta, utilizando:

```shell
python3 -m venv .
```

Então ative o ambiente virtual

```shell
source bin/activate
```

E, para finalizar a configuração, instale as dependências.

```shell
poetry install
```

Com o ambiente devidamente configurado, basta chamar o respectivo script. Ele deverá gerar todos os gráficos e tabelas (se houver) correspondentes.

```shell
python assignment1/trial_1.py
```
Ou:
```shell
python assignment1/trial_2.py
```
Aonde `assignment1/trial_1.py` representa a primeira parte do projeto e `assignment1/trial_2.py` representa a segunda parte.
