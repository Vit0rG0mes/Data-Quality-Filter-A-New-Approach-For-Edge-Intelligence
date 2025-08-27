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
from PIL import Image

# Configurações gerais
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Diretório de saída
output_dir = './generated_images_eurosat'
os.makedirs(output_dir, exist_ok=True)


# Parâmetros do treinamento
num_epochs = 20  # Manter baixo para teste inicial
batch_size = 64
learning_rate = 0.0002
image_size = 32  # Imagens serão redimensionadas para 32x32 para compatibilidade com a GAN
image_channels = 3
latent_dim = 100
num_classes_eurosat = 10 # EuroSAT tem 10 classes


# Função para calcular entropia (mantida)
def entropia(pk, base=2):
   # Garante que pk seja um array numpy
   if isinstance(pk, torch.Tensor):
       pk = pk.cpu().numpy()
   elif isinstance(pk, Image.Image):
       # Converte PIL Image para numpy array (assume RGB)
       pk = np.array(pk)

   # Calcula o histograma para imagens coloridas (considera cada canal)
   # Ou achata a imagem se for escala de cinza
   if pk.ndim == 3:
       # Achata os canais e dimensões espaciais para calcular um histograma único
       # Ou calcula entropia por canal e tira a média? Vamos achatar por simplicidade.
       pk_flat = pk.flatten()
       hist, _ = np.histogram(pk_flat, bins=256, range=(0, 256))
       pk = hist
   elif pk.ndim == 2:
       hist, _ = np.histogram(pk.flatten(), bins=256, range=(0, 256))
       pk = hist
   else:
       # Se já for um histograma ou similar, usa diretamente
       pass

   pk = pk / np.sum(pk) # Normaliza para obter distribuição de probabilidade
   pk = pk[pk > 0] # Remove zeros para evitar log(0)
   if len(pk) == 0:
       return 0.0 # Retorna 0 se não houver elementos > 0
   return -np.sum(pk * np.log(pk) / np.log(base))


# Função para filtrar os dados com base na entropia (adaptada para carregar imagens)
def filtrar_entropia_classe(dataset, totalClasses):
   train_Xextend, train_yextend = [], []
   # Agrupa índices por classe
   indices_por_classe = [[] for _ in range(totalClasses)]
   for idx, (_, label) in enumerate(dataset):
       indices_por_classe[label].append(idx)

   print(f"Calculando entropia e filtrando para {totalClasses} classes...")
   for label in range(totalClasses):
       indices_originais_classe = indices_por_classe[label]
       if not indices_originais_classe:
           print(f"Aviso: Nenhuma amostra encontrada para a classe {label}")
           continue

       # Calcula entropia para cada imagem da classe
       tuplas = []
       for index in indices_originais_classe:
           img, _ = dataset[index] # Carrega a imagem (PIL)
           # Converte PIL para numpy para a função entropia
           img_np = np.array(img)
           e = entropia(img_np)
           tuplas.append((index, e))

       if not tuplas:
           print(f"Aviso: Nenhuma entropia calculada para a classe {label}")
           continue

       local_ordenado = sorted(tuplas, key=lambda x: x[1])
       entropias = [item[1] for item in local_ordenado]
       if not entropias:
            print(f"Aviso: Lista de entropias vazia para a classe {label}")
            continue
       median = np.median(entropias)

       # Filtra índices com entropia <= mediana
       indices_filtrados_da_classe = [item[0] for item in local_ordenado if item[1] <= median]

       # Adiciona as imagens e rótulos filtrados (ainda como índices)
       # As imagens serão carregadas e transformadas pelo DataLoader depois
       train_Xextend.extend(indices_filtrados_da_classe) # Guarda os índices
       train_yextend.extend([label] * len(indices_filtrados_da_classe))
       print(f"Classe {label}: {len(indices_originais_classe)} originais -> {len(indices_filtrados_da_classe)} filtradas (mediana entropia: {median:.4f})")

   print(f"Total de imagens após filtragem: {len(train_Xextend)}")
   return train_Xextend, train_yextend


# Transformações de Data Augmentation (Adicionado Resize)
# Transformação para carregar e calcular entropia (sem normalização, sem ToTensor)
transform_entropy = transforms.Compose([
    transforms.Resize((64, 64)), # Redimensiona para tamanho original do EuroSAT para cálculo de entropia
])

# Transformação final para o DataLoader (com redimensionamento para GAN, augmentations, ToTensor, Normalize)
transform_train = transforms.Compose([
    transforms.Resize(image_size), # Redimensiona para 32x32
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(image_size, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Carregar o dataset EuroSAT (sem transformações iniciais, serão aplicadas depois)
print("Carregando EuroSAT dataset (pode levar um tempo para download na primeira vez)...")
# Primeiro, carregar para calcular entropia (usa transform_entropy)
eurosat_dataset_entropy = datasets.EuroSAT(root='./data_eurosat', download=True, transform=transform_entropy)
print(f"Dataset EuroSAT carregado para cálculo de entropia. Número de amostras: {len(eurosat_dataset_entropy)}")

# Filtrar os dados com base na entropia
filtered_indices, filtered_y = filtrar_entropia_classe(eurosat_dataset_entropy, totalClasses=num_classes_eurosat)

# Criar um novo dataset apenas com os índices filtrados e aplicar a transformação final
class FilteredEuroSAT(torch.utils.data.Dataset):
    def __init__(self, root, indices, targets, transform=None):
        # Recarregar o dataset base, mas sem transformações aqui
        self.base_dataset = datasets.EuroSAT(root=root, download=False, transform=None) # Download já deve ter ocorrido
        self.indices = indices
        self.targets = targets
        self.transform = transform
        # Mapear os índices originais para os novos índices (0 a N-1)
        self.original_to_filtered_idx = {original_idx: i for i, original_idx in enumerate(indices)}

    def __getitem__(self, index):
        original_idx = self.indices[index]
        img, _ = self.base_dataset[original_idx] # Pega a imagem original (PIL)
        target = self.targets[index]
        if self.transform:
            img = self.transform(img) # Aplica a transformação final
        return img, target

    def __len__(self):
        return len(self.indices)

print("Criando dataset filtrado com transformações finais...")
filtered_dataset = FilteredEuroSAT(root='./data_eurosat', indices=filtered_indices, targets=filtered_y, transform=transform_train)

# Verificar se o dataset filtrado não está vazio
if len(filtered_dataset) == 0:
    raise ValueError("O dataset filtrado está vazio. Verifique a lógica de filtragem ou o dataset original.")

filtered_data_loader = DataLoader(dataset=filtered_dataset, batch_size=batch_size, shuffle=True)
print(f"DataLoader para dataset filtrado criado. Número de batches: {len(filtered_data_loader)}")


# Definição do Gerador (mantido, compatível com 32x32)
class Generator(nn.Module):
   def __init__(self):
       super(Generator, self).__init__()
       self.model = nn.Sequential(
           # input is Z, going into a convolution
           nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
           nn.BatchNorm2d(512),
           nn.ReLU(True),
           # state size. (512) x 4 x 4
           nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
           nn.BatchNorm2d(256),
           nn.ReLU(True),
           # state size. (256) x 8 x 8
           nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
           nn.BatchNorm2d(128),
           nn.ReLU(True),
           # state size. (128) x 16 x 16
           nn.ConvTranspose2d( 128, image_channels, 4, 2, 1, bias=False),
           nn.Tanh()
           # state size. (image_channels) x 32 x 32
       )

   def forward(self, z):
       return self.model(z)


# Definição do Discriminador (mantido, compatível com 32x32)
class Discriminator(nn.Module):
   def __init__(self):
       super(Discriminator, self).__init__()
       self.model = nn.Sequential(
           # input is (image_channels) x 32 x 32
           nn.Conv2d(image_channels, 128, 4, 2, 1, bias=False),
           nn.LeakyReLU(0.2, inplace=True),
           # state size. (128) x 16 x 16
           nn.Conv2d(128, 256, 4, 2, 1, bias=False),
           nn.BatchNorm2d(256),
           nn.LeakyReLU(0.2, inplace=True),
           # state size. (256) x 8 x 8
           nn.Conv2d(256, 512, 4, 2, 1, bias=False),
           nn.BatchNorm2d(512),
           nn.LeakyReLU(0.2, inplace=True),
           # state size. (512) x 4 x 4
           nn.Conv2d(512, 1, 4, 1, 0, bias=False),
           nn.Sigmoid()
           # state size. 1
       )

   def forward(self, x):
       return self.model(x).view(-1, 1)

# Função de inicialização de pesos (recomendado para GANs)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Instanciar modelos e inicializar pesos
generator = Generator().to(device)
discriminator = Discriminator().to(device)
generator.apply(weights_init)
discriminator.apply(weights_init)

print("Modelos Gerador e Discriminador criados e inicializados.")

# Loss e otimizadores (mantidos)
criterion = nn.BCELoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))


# Inicializar variáveis para controle de tempo (mantido)
total_training_time = 0
epoch_times = []


# Função para formatar o tempo em horas:minutos:segundos (mantido)
def format_time(seconds):
   hours = int(seconds // 3600)
   minutes = int((seconds % 3600) // 60)
   seconds = int(seconds % 60)
   return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# Iniciar timer para o treinamento total
training_start_time = time.time()

# Transfer Learning setup with Inception v3 (mantido)
inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
inception_model.eval()
print("Modelo Inception v3 carregado para cálculo de Inception Score.")

# Add Inception Score calculation function (mantido)
def calculate_inception_score(images, model, n_split=10, eps=1e-16):
    model.eval()
    preds = []

    # Garante que as imagens estejam no device correto
    images = images.to(device)

    with torch.no_grad():
        for i in range(0, images.size(0), batch_size):
            batch = images[i:i+batch_size]
            # Redimensiona imagens para o tamanho de entrada do Inception
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            # Normalização específica do Inception v3 (se transform_input=False)
            # O modelo Inception_V3_Weights.DEFAULT já aplica a transformação necessária.
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

# Treinamento (lógica principal mantida)
avg_loss_g = []
avg_loss_d = []
accuracy_d_real = []
accuracy_d_fake = []
loss_g_values, loss_d_values = [], []

print(f"Iniciando treinamento por {num_epochs} épocas...")

fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device) # Ruído fixo para visualização

for epoch in range(num_epochs):
   epoch_start_time = time.time() # Timer para a época
   epoch_loss_g = 0.0
   epoch_loss_d = 0.0
   epoch_acc_real = 0.0
   epoch_acc_fake = 0.0
   num_batches = len(filtered_data_loader)

   for i, (real_images, _) in enumerate(filtered_data_loader):
       real_images = real_images.to(device)
       current_batch_size = real_images.size(0)

       # Labels reais e falsos
       label_real = torch.full((current_batch_size, 1), 1.0, dtype=torch.float, device=device)
       label_fake = torch.full((current_batch_size, 1), 0.0, dtype=torch.float, device=device)

       # --------------------- #
       #  Treinar Discriminador #
       # --------------------- #
       discriminator.zero_grad()

       # Loss com imagens reais
       output_real = discriminator(real_images)
       loss_real = criterion(output_real, label_real)
       loss_real.backward()
       D_x = output_real.mean().item()
       acc_real = (output_real >= 0.5).float().mean().item()

       # Loss com imagens falsas
       noise = torch.randn(current_batch_size, latent_dim, 1, 1, device=device)
       fake_images = generator(noise)
       output_fake = discriminator(fake_images.detach()) # detach() para não calcular gradientes no gerador
       loss_fake = criterion(output_fake, label_fake)
       loss_fake.backward()
       D_G_z1 = output_fake.mean().item()
       acc_fake = (output_fake < 0.5).float().mean().item()

       # Loss total do discriminador e passo do otimizador
       loss_d = loss_real + loss_fake
       optimizer_d.step()

       # ----------------- #
       #  Treinar Gerador  #
       # ----------------- #
       generator.zero_grad()
       output_fake_for_g = discriminator(fake_images) # Reutiliza fake_images, mas agora calcula gradientes
       loss_g = criterion(output_fake_for_g, label_real) # Gerador quer que D classifique fakes como reais
       loss_g.backward()
       D_G_z2 = output_fake_for_g.mean().item()
       optimizer_g.step()

       # Registrar perdas e acurácias do batch
       loss_g_values.append(loss_g.item())
       loss_d_values.append(loss_d.item())
       epoch_loss_g += loss_g.item()
       epoch_loss_d += loss_d.item()
       epoch_acc_real += acc_real
       epoch_acc_fake += acc_fake

       if (i + 1) % 100 == 0:
           print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{num_batches}], Loss_D: {loss_d.item():.4f}, Loss_G: {loss_g.item():.4f}, D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}, Acc Real: {acc_real:.4f}, Acc Fake: {acc_fake:.4f}')

   # Fim da Época
   epoch_end_time = time.time()
   epoch_time = epoch_end_time - epoch_start_time
   epoch_times.append(epoch_time)
   total_training_time += epoch_time

   # Calcular médias da época
   avg_loss_g_epoch = epoch_loss_g / num_batches
   avg_loss_d_epoch = epoch_loss_d / num_batches
   avg_acc_real_epoch = epoch_acc_real / num_batches
   avg_acc_fake_epoch = epoch_acc_fake / num_batches
   avg_loss_g.append(avg_loss_g_epoch)
   avg_loss_d.append(avg_loss_d_epoch)
   accuracy_d_real.append(avg_acc_real_epoch)
   accuracy_d_fake.append(avg_acc_fake_epoch)

   print(f"--- Fim Epoch [{epoch+1}/{num_epochs}] --- Tempo: {format_time(epoch_time)} --- ")
   print(f"Média Loss_G: {avg_loss_g_epoch:.4f}, Média Loss_D: {avg_loss_d_epoch:.4f}")
   print(f"Média Acc Real: {avg_acc_real_epoch:.4f}, Média Acc Fake: {avg_acc_fake_epoch:.4f}")

   # Gerar e salvar imagens de exemplo
   with torch.no_grad():
      generator.eval() # Modo de avaliação para gerar imagens
      fake_samples = generator(fixed_noise).detach().cpu()
      generator.train() # Voltar ao modo de treinamento

      fake_grid = torchvision.utils.make_grid(fake_samples, padding=2, normalize=True)
      plt.figure(figsize=(8, 8))
      plt.imshow(np.transpose(fake_grid, (1, 2, 0)))
      plt.title(f'Época {epoch+1}')
      plt.axis('off')
      image_path = os.path.join(output_dir, f'epoch_{epoch+1:03d}.png')
      plt.savefig(image_path, bbox_inches='tight')
      # plt.show() # Descomente se quiser exibir interativamente
      plt.close()
      print(f"Imagens de exemplo salvas em: {image_path}")

   # Calcular Inception Score (a cada 5 épocas ou na última)
   if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
     print("Calculando Inception Score...")
     with torch.no_grad():
       generator.eval()
       # Gerar mais amostras para IS
       eval_noise = torch.randn(1000, latent_dim, 1, 1, device=device)
       eval_samples = generator(eval_noise).detach()
       generator.train()
       is_score, is_std = calculate_inception_score(eval_samples, inception_model)
       inception_scores.append(is_score)
       print(f'--- Epoch [{epoch+1}/{num_epochs}] - Inception Score: {is_score:.4f} ± {is_std:.4f} ---')

# Fim do Treinamento
print(f"Treinamento concluído em {format_time(total_training_time)}.")

# Calcular a média dos Inception Scores ao final do treinamento
if inception_scores:
    average_inception_score = sum(inception_scores) / len(inception_scores)
    print(f'Média final do Inception Score: {average_inception_score:.4f}')
else:
    print('Nenhum Inception Score foi calculado (verifique a frequência de cálculo e número de épocas).')


# Plotar as perdas
plt.figure(figsize=(12, 6))
plt.plot(avg_loss_g, label='Loss Média Gerador (por Época)')
plt.plot(avg_loss_d, label='Loss Média Discriminador (por Época)')
plt.xlabel('Época')
plt.ylabel('Loss Média')
plt.title('Curvas de Loss Médio por Época - Gerador e Discriminador (EuroSAT Filtrado)')
plt.legend()
plt.grid(True)

# Adicionar informações ao gráfico
info_text = (
    f'Tempo Total: {format_time(total_training_time)}\n'
    f'Épocas: {num_epochs}\n'
    f'Batch Size: {batch_size}\n'
    f'LR: {learning_rate}\n'
    f'Imgs Filtradas: {len(filtered_dataset)}\n'
    f'Média Final Loss G: {avg_loss_g[-1]:.4f}\n'
    f'Média Final Loss D: {avg_loss_d[-1]:.4f}\n'
    f'Média Final Acc Real: {accuracy_d_real[-1]:.4f}\n'
    f'Média Final Acc Fake: {accuracy_d_fake[-1]:.4f}\n'
    f'Média Final IS: {average_inception_score:.4f}' if inception_scores else f'Média Final IS: N/A'
)
plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
         fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

plt.tight_layout()
loss_curve_path = 'curvas_de_loss_eurosat_filtrado.png'
plt.savefig(loss_curve_path, dpi=300)
print(f"Gráfico de curvas de loss salvo em: {loss_curve_path}")
plt.close()

print("Script finalizado.")
