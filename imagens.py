from pybrain.datasets.supervised import SupervisedDataSet 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

# processamento de imagens
from PIL import Image

# gráficos 
import matplotlib.pyplot as plt
import numpy as np

# função para carregar os dados de treinameto a partir das imagens
def getDataImage( path):
    #Read image
    img = Image.open( path )
    # create the pixel map
    pixels = img.load() 
    data = []
    pixel = []
    for i in range( img.size[0]):         # for every col do: img.size[0]
        for j in range( img.size[1] ):    # for every row   img.size[1]      
             pixel = pixels[i,j]          # get every pixel
             data.append( pixel[0] )
             data.append( pixel[1] )
             data.append( pixel[2] )

    #Viewing EXIF data embedded in image
    exif_data = img._getexif()
    exif_data
    return data

# carregando a primeira imagem
dataTraining =  getDataImage( 'img\\lub1.jpeg' )
size = 89 * 50 * 3

# configurando a rede neural artificial e o dataSet de treinamento
network = buildNetwork( size, 100, 20, 4 )  # define network
dataSet = SupervisedDataSet( size, 4 )      # define dataSet


# load dataSet
dataSet.addSample ( getDataImage( 'img\\lub1.jpeg' ), (1, 1, 1, 1) )       # lubrificada
dataSet.addSample ( getDataImage( 'img\\lub2.jpeg' ), (1, 1, 1, 1) )       #lubrificada
dataSet.addSample ( getDataImage( 'img\\lub3.jpeg' ), (1, 1, 1, 1) )       # lubrificada
dataSet.addSample ( getDataImage( 'img\\naoLub1.jpeg' ), (0, 0, 0, 0) )    # naoLubrificada
dataSet.addSample ( getDataImage( 'img\\naoLub2.jpeg' ),(0, 0, 0, 0) )     # naoLubrificada
dataSet.addSample ( getDataImage( 'img\\naoLub3.jpeg' ), (0, 0, 0, 0) )    # naoLubrificada


# trainer
trainer = BackpropTrainer( network, dataSet)
error = 1
iteration = 0
outputs = []
file = open("outputs.txt", "w")

while error > 0.001: 
    error = trainer.train()
    outputs.append( error )
    iteration += 1    
    print ( iteration, error )
    file.write( str(error)+"\n" )

file.close()

# plot graph
plt.ioff()
plt.plot( outputs )
plt.xlabel('Iterações')
plt.ylabel('Erro Quadrático')
plt.show()

# Fase de teste
name = ['teste1.jpeg', 'teste2.jpeg'] # teste1 = não lubrificada teste2 = lubrificada
for i in range( len(name) ):
    path = "img\\test\\" + name[i]
    print ( path )
    print ( network.activate( getDataImage( path ) ) )