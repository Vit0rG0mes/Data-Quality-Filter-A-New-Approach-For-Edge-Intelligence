# Importações
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision.models import inception_v3, Inception_V3_Weights
import torch.nn.functional as F


torch.manual_seed(42)
np.random.seed(42)

import os
# Criar um diretório para salvar as imagens
output_dir = './generated_images'
os.makedirs(output_dir, exist_ok=True)
# Nome do arquivo de log e preparação para registrar o log.
log_file = 'training_log.txt'
with open(log_file, 'w') as f:
    f.write("")

# Parâmetros do treinamento
num_epochs = 20
batch_size = 64
learning_rate = 0.0002
image_size = 28
image_channels = 1
latent_dim = 100
total_classes = 10

#Data Augmentation
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Carregamento do dataset MNIST
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True)

# Filtrar dados com base em entropia
train_X = mnist_dataset.data.numpy()  # Imagens
train_y = mnist_dataset.targets.numpy()  # Rótulos

# Criar DataLoader normalizado
normalizeData = TensorDataset(
    torch.tensor(train_X, dtype=torch.float32).unsqueeze(1) / 255.0,  # Normaliza
    torch.tensor(train_y, dtype=torch.long)
)
data_loader = DataLoader(dataset=normalizeData, batch_size=batch_size, shuffle=True)

# Definição do Gerador
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential( #dimensao do vetor de entrada, numero de filtros/canais da rede, 7 e numero de kernel, stride, paggin
            nn.ConvTranspose2d(latent_dim, 512, 7, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),


            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.ConvTranspose2d(64, image_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Definição do Discriminador
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),

            nn.Conv2d(512, 1, 7, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Instanciar modelos
generator = Generator().cuda()
discriminator = Discriminator().cuda()

# Loss e Otimizadores
criterion = nn.BCELoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

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

            # Replicate the grayscale channel to create a 3-channel image
            batch = batch.repeat(1, 3, 1, 1)  # Repeat grayscale channel 3 times

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
loss_g_values = []
loss_d_values = []
avg_loss_g = []
avg_loss_d = []
accuracy_d = []
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(data_loader):
        real_images = real_images.cuda()
        batch_size = real_images.size(0)

        # Treinamento do Discriminador
        optimizer_d.zero_grad()
        label_real = torch.ones(batch_size, 1).cuda()
        output_real = discriminator(real_images).view(-1, 1)
        loss_real = criterion(output_real, label_real)
        loss_real.backward()

        noise = torch.randn(batch_size, latent_dim, 1, 1).cuda()
        fake_images = generator(noise)
        label_fake = torch.zeros(batch_size, 1).cuda()
        output_fake = discriminator(fake_images.detach()).view(-1, 1)
        loss_fake = criterion(output_fake, label_fake)
        loss_fake.backward()
        optimizer_d.step()

        # Treinamento do Gerador
        optimizer_g.zero_grad()
        output = discriminator(fake_images).view(-1, 1)
        loss_g = criterion(output, label_real)
        loss_g.backward()
        optimizer_g.step()

        # Registrar perdas
        loss_g_values.append(loss_g.item())
        loss_d_values.append(loss_real.item() + loss_fake.item())

        if (i + 1) % 100 == 0:
            log_message = (f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(data_loader)}], '
                           f'D_real: {output_real.mean():.4f}, D_fake: {output_fake.mean():.4f}, '
                           f'Loss_D: {loss_real.item() + loss_fake.item():.4f}, Loss_G: {loss_g.item():.4f}')

            # Imprimir o log no console
            print(log_message)

            # Salvar o log no arquivo
            with open(log_file, 'a') as f:
                f.write(log_message + "\n")
    # Média das perdas por época
    avg_loss_g.append(np.mean(loss_g_values[-len(data_loader):]))
    avg_loss_d.append(np.mean(loss_d_values[-len(data_loader):]))

    # Cálculo da acurácia do Discriminador
    correct_real = torch.sum(output_real > 0.5).item()
    correct_fake = torch.sum(output_fake < 0.5).item()
    accuracy = (correct_real + correct_fake) / (2 * batch_size)
    accuracy_d.append(accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss_G: {avg_loss_g[-1]:.4f}, Loss_D: {avg_loss_d[-1]:.4f}, Accuracy_D: {accuracy:.4f}")

    #Gerar e salvar imagens de amostra no final de cada época
    with torch.no_grad():
        fake_samples = generator(torch.randn(64, latent_dim, 1, 1).cuda())
        fake_samples = fake_samples.cpu()
        fake_grid = torchvision.utils.make_grid(fake_samples, padding=2, normalize=True)

        # Configurar o plot
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(fake_grid, (1, 2, 0)))
        plt.axis('off')
        plt.show()

        # Salvar a imagem no diretório especificado
        image_path = os.path.join(output_dir, f'epoch_{epoch+1:03d}.png')
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()

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


# Salvar o modelo do gerador treinado
torch.save(generator.state_dict(), 'generator.pth')

# Plotar as perdas
plt.figure(figsize=(16, 7))
plt.plot(loss_g_values, label='Loss Gerador')
plt.plot(loss_d_values, label='Loss Discriminador')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Curves - Gerador e Discriminador')

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
plt.show()
