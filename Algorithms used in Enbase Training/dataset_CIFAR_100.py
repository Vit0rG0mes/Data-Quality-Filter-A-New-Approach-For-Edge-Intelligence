import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights
import torch.nn.functional as F
import torchvision
import os

# Configurações gerais
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Diretório de saída
output_dir = './generated_images'
os.makedirs(output_dir, exist_ok=True)


# Parâmetros do treinamento
num_epochs = 1
batch_size = 64
learning_rate = 0.0002
image_size = 32
image_channels = 3
latent_dim = 100


# Função para calcular entropia
def entropia(pk, base=2):
   pk = pk / np.sum(pk)
   pk = pk[pk > 0]
   return -np.sum(pk * np.log(pk) / np.log(base))


# Função para filtrar os dados com base na entropia
def filtrar_entropia_classe(train_X, train_y, totalClasses):
   train_Xextend, train_yextend = [], []
   for label in range(totalClasses):
       indices_originais = np.where(train_y == label)[0]
       indicesDasImagensDaClasse = train_X[indices_originais]
       tuplas = [(indices_originais[index], entropia(img)) for index, img in enumerate(indicesDasImagensDaClasse)]
       local_ordenado = sorted(tuplas, key=lambda x: x[1])
       median = np.median([item[1] for item in local_ordenado])
       indices_filtrados_da_classe = [item[0] for item in local_ordenado if item[1] <= median]
       train_Xextend.extend([train_X[i] for i in indices_filtrados_da_classe])
       train_yextend.extend([train_y[i] for i in indices_filtrados_da_classe])
   return np.array(train_Xextend), np.array(train_yextend)


# Transformações de Data Augmentation
transform = transforms.Compose([
   transforms.RandomHorizontalFlip(),
   transforms.RandomCrop(image_size, padding=4),
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Carregar o dataset CIFAR-100
cifar100_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=cifar100_dataset, batch_size=batch_size, shuffle=True)


# Extrair imagens e rótulos
train_X = np.transpose(cifar100_dataset.data, (0, 3, 1, 2))
train_y = np.array(cifar100_dataset.targets)


# Filtrar os dados com base na entropia
filtered_X, filtered_y = filtrar_entropia_classe(train_X, train_y, totalClasses=100)
filtered_dataset = TensorDataset(
   torch.tensor(filtered_X, dtype=torch.float32) / 255.0,
   torch.tensor(filtered_y, dtype=torch.long)
)
filtered_data_loader = DataLoader(dataset=filtered_dataset, batch_size=batch_size, shuffle=True)


# Definição do Gerador
class Generator(nn.Module):
   def __init__(self):
       super(Generator, self).__init__()
       self.model = nn.Sequential(
           nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
           nn.BatchNorm2d(512),
           nn.ReLU(True),
           nn.ConvTranspose2d(512, 256, 4, 2, 1),
           nn.BatchNorm2d(256),
           nn.ReLU(True),
           nn.ConvTranspose2d(256, 128, 4, 2, 1),
           nn.BatchNorm2d(128),
           nn.ReLU(True),
           nn.ConvTranspose2d(128, image_channels, 4, 2, 1),
           nn.Tanh()
       )


   def forward(self, z):
       return self.model(z)


# Definição do Discriminador
class Discriminator(nn.Module):
   def __init__(self):
       super(Discriminator, self).__init__()
       self.model = nn.Sequential(
           nn.Conv2d(image_channels, 128, 4, 2, 1),
           nn.LeakyReLU(0.2, inplace=True),
           nn.Conv2d(128, 256, 4, 2, 1),
           nn.BatchNorm2d(256),
           nn.LeakyReLU(0.2, inplace=True),
           nn.Conv2d(256, 512, 4, 2, 1),
           nn.BatchNorm2d(512),
           nn.LeakyReLU(0.2, inplace=True),
           nn.Conv2d(512, 1, 4, 1, 0),
           nn.Sigmoid()
       )


   def forward(self, x):
       return self.model(x)


# Instanciar modelos
generator = Generator().to(device)
discriminator = Discriminator().to(device)


# Loss e otimizadores
criterion = nn.BCELoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))


# Inicializar variáveis para controle de tempo
total_training_time = 0
epoch_times = []


# Função para formatar o tempo em horas:minutos:segundos
def format_time(seconds):
   hours = int(seconds // 3600)
   minutes = int((seconds % 3600) // 60)
   seconds = int(seconds % 60)
   return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# Iniciar timer para o treinamento total
training_start_time = time.time()


# Iniciar timer para a primeira época
epoch_start_time = time.time()

# Transfer Learning setup with Inception v3
inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).cuda()
inception_model.eval()

# Add Inception Score calculation function
def calculate_inception_score(images, model, n_split=10, eps=1e-16):
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, images.size(0), batch_size):
            batch = images[i:i+batch_size].cuda()
            # Resize images to inception input size
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear')
            pred = model(batch)
            preds.append(F.softmax(pred, dim=1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    scores = []

    for i in range(n_split):
        part = preds[(i * preds.shape[0] // n_split):((i + 1) * preds.shape[0] // n_split), :]
        kl = part * (np.log(part + eps) - np.log(np.expand_dims(np.mean(part, 0), 0) + eps))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)

# Add inception score tracking
inception_scores = []

# Treinamento
avg_loss_g = []
avg_loss_d = []
accuracy_d = []
loss_g_values, loss_d_values = [], []
for epoch in range(num_epochs):
   for i, (real_images, _) in enumerate(filtered_data_loader):
       real_images = real_images.to(device)
       batch_size = real_images.size(0)

       # Treinar o Discriminador
       optimizer_d.zero_grad()
       label_real = torch.ones(batch_size, 1).to(device)
       label_fake = torch.zeros(batch_size, 1).to(device)

       output_real = discriminator(real_images).view(-1, 1)
       loss_real = criterion(output_real, label_real)

       noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
       fake_images = generator(noise)
       output_fake = discriminator(fake_images.detach()).view(-1, 1)
       loss_fake = criterion(output_fake, label_fake)

       loss_d = loss_real + loss_fake
       loss_d.backward()
       optimizer_d.step()


       # Treinar o Gerador
       optimizer_g.zero_grad()
       output = discriminator(fake_images).view(-1, 1)
       loss_g = criterion(output, label_real)
       loss_g.backward()
       optimizer_g.step()


       # Registrar perdas
       loss_g_values.append(loss_g.item())
       loss_d_values.append(loss_d.item())
       if (i + 1) % 100 == 0:
           log_message = (f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(data_loader)}], '
                           f'D_real: {output_real.mean():.4f}, D_fake: {output_fake.mean():.4f}, '
                           f'Loss_D: {loss_real.item() + loss_fake.item():.4f}, Loss_G: {loss_g.item():.4f}')

           # Imprimir log no console
           print(log_message)


   print(f"Epoch [{epoch+1}/{num_epochs}], Loss_D: {loss_d.item():.4f}, Loss_G: {loss_g.item():.4f}")
   with torch.no_grad():
      fake_samples = generator(torch.randn(64, latent_dim, 1, 1).cuda())
      fake_samples = fake_samples.cpu()
      fake_grid = torchvision.utils.make_grid(fake_samples, padding=2, normalize=True)

      # Configurar o plot
      plt.figure(figsize=(8, 8))
      plt.imshow(np.transpose(fake_grid, (1, 2, 0)))
      plt.axis('off')

      # Salvar a imagem no diretório especificado
      image_path = os.path.join(output_dir, f'epoch_{epoch+1:03d}.png')
      plt.savefig(image_path, bbox_inches='tight')

      # Exibir a imagem
      plt.show()

      # Fechar a figura para liberar memória
      plt.close()

   # Média das perdas por época
   avg_loss_g.append(np.mean(loss_g_values[-len(filtered_data_loader):]))
   avg_loss_d.append(np.mean(loss_d_values[-len(filtered_data_loader):]))

      # Cálculo da acurácia do Discriminador
   correct_real = torch.sum(output_real > 0.5).item()
   correct_fake = torch.sum(output_fake < 0.5).item()
   accuracy = (correct_real + correct_fake) / (2 * batch_size)
   accuracy_d.append(accuracy)

   print(f"Epoch [{epoch+1}/{num_epochs}] - Loss_G: {avg_loss_g[-1]:.4f}, Loss_D: {avg_loss_d[-1]:.4f}, Accuracy_D: {accuracy:.4f}")

# Atualizar o timer para a época atual
   epoch_end_time = time.time()
   epoch_time = epoch_end_time - epoch_start_time
   epoch_times.append(epoch_time)
   epoch_start_time = epoch_end_time
   total_training_time += epoch_time

   # Calculate Inception Score
   if epoch % 5 == 0 or epoch == num_epochs - 1:  # Calculate every 5 epochs
     with torch.no_grad():
       eval_samples = generator(torch.randn(1000, latent_dim, 1, 1).cuda())
       is_score, is_std = calculate_inception_score(eval_samples, inception_model)
       inception_scores.append(is_score)
       print(f'Epoch [{epoch+1}/{num_epochs}] - Inception Score: {is_score:.4f} ± {is_std:.4f}')

# Calcular a média dos Inception Scores ao final do treinamento
if inception_scores:  # Verifica se a lista não está vazia
    average_inception_score = sum(inception_scores) / len(inception_scores)
    print(f'Média do Inception Score ao final do treinamento: {average_inception_score:.4f}')
else:
    print('Nenhum Inception Score foi calculado.')


# Plotar as perdas
plt.figure(figsize=(16, 7))
plt.plot(loss_g_values, label='Loss Gerador')
plt.plot(loss_d_values, label='Loss Discriminador')
plt.xlabel('Iterações')
plt.ylabel('Loss')
plt.title('Curvas de Loss - Gerador e Discriminador')

plt.text(0.02, 0.98, f'Tempo Total: {format_time(total_training_time)}',
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

plt.text(0.02, 0.7, f'Média Loss Gerador: {sum(avg_loss_g)/len(avg_loss_g):.4f}',
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8),
         verticalalignment='top')

plt.text(0.02, 0.8, f'Média Loss Discriminador: {sum(avg_loss_d)/len(avg_loss_d):.4f}',
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8),
         verticalalignment='top')

plt.text(0.02, 0.9, f'Média Acurácia: {sum(accuracy_d)/len(accuracy_d):.4f}',
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8),
         verticalalignment='top')

plt.legend()
plt.grid()
plt.savefig('curvas_de_loss.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
