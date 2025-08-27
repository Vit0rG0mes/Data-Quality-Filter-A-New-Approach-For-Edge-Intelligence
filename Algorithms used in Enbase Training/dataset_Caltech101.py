import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights
import torch.nn.functional as F
import torchvision
import os
from PIL import Image
from tqdm import tqdm
import requests
import zipfile
import shutil
import glob
import tarfile

# Configurações gerais
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Diretório de saída
output_dir = './generated_images_caltech101'
os.makedirs(output_dir, exist_ok=True)

# Parâmetros do treinamento
num_epochs = 20
batch_size = 32  # Reduzido para economizar memória
learning_rate = 0.0002
image_size = 64  # Reduzido para economizar memória (originalmente 128)
image_channels = 3
latent_dim = 100
num_classes = 102  # Caltech101 tem 101 classes + 1 background

# Função para baixar e extrair o dataset Caltech101 manualmente
def download_caltech101(root_dir):
    """
    Baixa e extrai o dataset Caltech101 manualmente, já que o download automático
    via torchvision pode falhar.
    """
    # URL alternativa para o dataset Caltech101
    url = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"

    # Cria o diretório se não existir
    os.makedirs(root_dir, exist_ok=True)

    # Caminho para o arquivo zip
    zip_path = os.path.join(root_dir, "caltech101.zip")

    # Caminho para o diretório extraído
    extract_path = os.path.join(root_dir, "caltech101")

    # Caminho para o diretório final esperado
    final_path = os.path.join(extract_path, "101_ObjectCategories")

    # Verifica se o dataset já foi baixado e extraído corretamente
    if os.path.exists(final_path) and len(os.listdir(final_path)) > 0:
        print(f"Dataset Caltech101 já existe em {final_path}")
        return final_path

    # Baixa o arquivo se não existir
    if not os.path.exists(zip_path):
        print(f"Baixando Caltech101 de {url}...")
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(zip_path, 'wb') as file:
                for data in tqdm(response.iter_content(chunk_size=1024),
                                total=total_size//1024, unit='KB'):
                    file.write(data)

            print(f"Download concluído: {zip_path}")
        except Exception as e:
            print(f"Erro ao baixar o dataset: {e}")
            return None

    # Cria o diretório de extração se não existir
    os.makedirs(extract_path, exist_ok=True)

    # Extrai o arquivo zip
    print(f"Extraindo {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root_dir)
        print(f"Extração do zip concluída")
    except Exception as e:
            print(f"Erro ao extrair o arquivo zip: {e}")
            return None

    # Verifica se o arquivo 101_ObjectCategories.tar.gz existe
    tar_path = None
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "101_ObjectCategories.tar.gz":
                tar_path = os.path.join(root, file)
                break
        if tar_path:
            break

    if not tar_path:
        print("Arquivo 101_ObjectCategories.tar.gz não encontrado após extração do zip")
        return None

    # Extrai o arquivo tar.gz
    print(f"Extraindo {tar_path}...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_path)
        print(f"Extração do tar.gz concluída")
    except Exception as e:
        print(f"Erro ao extrair o arquivo tar.gz: {e}")
        return None


    # Verifica se o diretório 101_ObjectCategories foi criado
    if not os.path.exists(final_path):
        # Procura pelo diretório 101_ObjectCategories em qualquer lugar após a extração
        found_path = None
        for root, dirs, files in os.walk(root_dir):
            if "101_ObjectCategories" in dirs:
                found_path = os.path.join(root, "101_ObjectCategories")
                break

        if found_path and found_path != final_path:
            print(f"Movendo diretório de {found_path} para {final_path}")
            # Se o diretório de destino já existe, remova-o primeiro
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            # Mova o diretório encontrado para o local esperado
            shutil.move(found_path, final_path)


    # Verifica novamente se o diretório final existe
    if not os.path.exists(final_path):
        print(f"Diretório {final_path} não encontrado após extração")
        return None


    print(f"Dataset preparado com sucesso em: {final_path}")
    return final_path


# Classe para o dataset Caltech101 personalizado
class Caltech101Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # Verifica se o diretório existe
        if not os.path.exists(root):
            raise RuntimeError(f"Diretório do dataset não encontrado: {root}")

        # Lista as classes (diretórios)
        try:
            self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
            if not self.classes:
                raise RuntimeError(f"Nenhuma classe (diretório) encontrada em {root}")

            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

            self.samples = []
            for class_name in self.classes:
                class_dir = os.path.join(root, class_name)
                for img_path in glob.glob(os.path.join(class_dir, "*.jpg")):
                    self.samples.append((img_path, self.class_to_idx[class_name]))

            if not self.samples:
                raise RuntimeError(f"Nenhuma imagem encontrada em {root}")

            print(f"Carregado dataset com {len(self.samples)} imagens em {len(self.classes)} classes")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar o dataset: {e}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')

            if self.transform:
                img = self.transform(img)

            return img, label
        except Exception as e:
            print(f"Erro ao carregar imagem {img_path}: {e}")
            # Retorna uma imagem preta em caso de erro
            if self.transform:
                return self.transform(Image.new('RGB', (image_size, image_size), (0, 0, 0))), label
            else:
                return Image.new('RGB', (image_size, image_size), (0, 0, 0)), label


# Função para calcular entropia
def entropia(pk, base=2):
    """
    Calcula a entropia de uma distribuição de probabilidade.
    Adaptada para lidar com diferentes formatos de entrada.
    """
    # Converte para array numpy se for tensor
    if isinstance(pk, torch.Tensor):
        pk = pk.cpu().numpy()

    # Se for uma imagem PIL, converte para array
    if isinstance(pk, Image.Image):
        pk = np.array(pk)

    # Para imagens, calcula o histograma
    if pk.ndim == 3:  # Imagem colorida
        # Achata a imagem e calcula o histograma
        pk_flat = pk.flatten()
        hist, _ = np.histogram(pk_flat, bins=256, range=(0, 255))
        pk = hist

    # Normaliza para obter uma distribuição de probabilidade
    pk = pk / np.sum(pk)
    pk = pk[pk > 0]  # Remove zeros para evitar log(0)

    # Retorna 0 se não houver elementos positivos
    if len(pk) == 0:
        return 0.0

    return -np.sum(pk * np.log(pk) / np.log(base))

# Classe para o dataset Caltech101 com filtragem por entropia
class Caltech101Filtered(Dataset):
    def __init__(self, root, transform=None, download=False, filter_by_entropy=True):
        # Baixa o dataset se necessário
        if download:
            dataset_path = download_caltech101(root)
            if dataset_path:
                self.data_dir = dataset_path
            else:
                raise RuntimeError("Falha ao baixar ou extrair o dataset Caltech101")
        else:
            # Tenta encontrar o diretório do dataset
            possible_paths = [
                os.path.join(root, "caltech101", "101_ObjectCategories"),
                os.path.join(root, "101_ObjectCategories")
            ]

            self.data_dir = None
            for path in possible_paths:
                if os.path.exists(path) and os.path.isdir(path):
                    self.data_dir = path
                    break

            if not self.data_dir:
                raise RuntimeError(f"Dataset não encontrado. Tente usar download=True ou verifique os caminhos: {possible_paths}")

        print(f"Usando dataset em: {self.data_dir}")

        # Carrega o dataset original
        try:
            self.original_dataset = Caltech101Dataset(
                root=self.data_dir,
                transform=None  # Sem transformação inicial para calcular entropia
            )
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar o dataset original: {e}")


        self.transform = transform

        # Filtra por entropia se solicitado
        if filter_by_entropy:
            self.filtered_indices, self.filtered_targets = self.filter_by_entropy()
        else:
            # Usa todos os índices se não filtrar
            self.filtered_indices = list(range(len(self.original_dataset)))
            self.filtered_targets = [self.original_dataset.samples[i][1] for i in self.filtered_indices]

    def filter_by_entropy(self):
        """
        Filtra as imagens com base na entropia, mantendo apenas aquelas
        com entropia menor ou igual à mediana para cada classe.
        """
        print("Calculando entropia e filtrando imagens...")

        # Agrupa índices por classe
        indices_por_classe = {}
        for idx, (_, label) in enumerate(self.original_dataset.samples):
            if label not in indices_por_classe:
                indices_por_classe[label] = []
            indices_por_classe[label].append(idx)

        filtered_indices = []
        filtered_targets = []

        # Para cada classe, calcula a entropia e filtra
        for label, indices in tqdm(indices_por_classe.items()):
            # Calcula entropia para cada imagem da classe
            entropias = []
            for idx in indices:
                try:
                    img, _ = self.original_dataset[idx]
                    img_np = np.array(img)
                    e = entropia(img_np)
                    entropias.append((idx, e))
                except Exception as e:
                    print(f"Erro ao processar imagem {idx}: {e}")
                    # Pula esta imagem
                    continue

            # Ordena por entropia e pega a mediana
            entropias.sort(key=lambda x: x[1])
            if entropias:  # Verifica se a lista não está vazia
                median_idx = len(entropias) // 2
                median_entropy = entropias[median_idx][1]

                # Filtra índices com entropia <= mediana
                filtered_class_indices = [idx for idx, e in entropias if e <= median_entropy]

                # Adiciona os índices e rótulos filtrados
                filtered_indices.extend(filtered_class_indices)
                filtered_targets.extend([label] * len(filtered_class_indices))

                print(f"Classe {label}: {len(indices)} originais -> {len(filtered_class_indices)} filtradas (mediana entropia: {median_entropy:.4f})")


        print(f"Total de imagens após filtragem: {len(filtered_indices)} de {len(self.original_dataset)}")
        return filtered_indices, filtered_targets

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # Obtém o índice original
        original_idx = self.filtered_indices[idx]
        # Carrega a imagem e o rótulo
        img, _ = self.original_dataset[original_idx]
        target = self.filtered_targets[idx]

        # Aplica transformações se necessário
        if self.transform:
            img = self.transform(img)

        return img, target

# Transformações para o dataset
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # Redimensiona para tamanho padrão
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(image_size, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Carrega o dataset Caltech101
print("Carregando Caltech101 dataset...")
data_dir = './data'
try:
    caltech101_dataset = Caltech101Filtered(
        root=data_dir,
        transform=transform,
        download=True,  # Sempre tenta baixar para garantir
        filter_by_entropy=True
    )

    # Cria o DataLoader
    filtered_data_loader = DataLoader(
        dataset=caltech101_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2  # Reduzido para economizar recursos
    )
except Exception as e:
    print(f"Erro ao carregar o dataset: {e}")
    raise


# Definição do Gerador
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. (512) x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (256) x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (128) x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (64) x 32 x 32
            nn.ConvTranspose2d(64, image_channels, 4, 2, 1),
            nn.Tanh()
            # state size. (image_channels) x 64 x 64
        )

    def forward(self, z):
        return self.model(z)

# Definição do Discriminador
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # input is (image_channels) x 64 x 64
            nn.Conv2d(image_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (128) x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (256) x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (512) x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)

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

# Transfer Learning setup with Inception v3
print("Carregando modelo Inception v3 para cálculo de Inception Score...")
try:
    inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
    inception_model.eval()
except Exception as e:
    print(f"Erro ao carregar o modelo Inception v3: {e}")
    print("Continuando sem cálculo de Inception Score")
    inception_model = None

# Add Inception Score calculation function
def calculate_inception_score(images, model, n_split=10, eps=1e-16):
    if model is None:
        print("Modelo Inception não disponível, pulando cálculo de Inception Score")
        return 0.0, 0.0

    model.eval()
    preds = []

    try:
        with torch.no_grad():
            for i in range(0, images.size(0), batch_size):
                batch = images[i:i+batch_size].to(device)
                # Resize images to inception input size
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
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
    except Exception as e:
        print(f"Erro ao calcular Inception Score: {e}")
        return 0.0, 0.0

# Add inception score tracking
inception_scores = []

# Treinamento
avg_loss_g = []
avg_loss_d = []
accuracy_d = []
loss_g_values, loss_d_values = [], []

# Iniciar timer para a primeira época
epoch_start_time = time.time()

try:
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(filtered_data_loader):
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)

            # Treinar o Discriminador
            optimizer_d.zero_grad()
            label_real = torch.ones(current_batch_size, 1).to(device)
            label_fake = torch.zeros(current_batch_size, 1).to(device)

            output_real = discriminator(real_images).view(-1, 1)
            loss_real = criterion(output_real, label_real)

            noise = torch.randn(current_batch_size, latent_dim, 1, 1).to(device)
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
            if (i + 1) % 50 == 0:  # Reduzido para economizar saída
                log_message = (f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(filtered_data_loader)}], '
                            f'D_real: {output_real.mean():.4f}, D_fake: {output_fake.mean():.4f}, '
                            f'Loss_D: {loss_real.item() + loss_fake.item():.4f}, Loss_G: {loss_g.item():.4f}')

                # Imprimir log no console
                print(log_message)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss_D: {loss_d.item():.4f}, Loss_G: {loss_g.item():.4f}")
        with torch.no_grad():
            fake_samples = generator(torch.randn(64, latent_dim, 1, 1).to(device))
            fake_samples = fake_samples.cpu()
            fake_grid = torchvision.utils.make_grid(fake_samples, padding=2, normalize=True)

            # Configurar o plot
            plt.figure(figsize=(8, 8))
            plt.imshow(np.transpose(fake_grid, (1, 2, 0)))
            plt.axis('off')

            # Salvar a imagem no diretório especificado
            image_path = os.path.join(output_dir, f'epoch_{epoch+1:03d}.png')
            plt.savefig(image_path, bbox_inches='tight')
            print(f"Imagem de exemplo salva em: {image_path}")

            # Fechar a figura para liberar memória
            plt.close()

        # Média das perdas por época
        avg_loss_g.append(np.mean(loss_g_values[-len(filtered_data_loader):]))
        avg_loss_d.append(np.mean(loss_d_values[-len(filtered_data_loader):]))

        # Cálculo da acurácia do Discriminador
        correct_real = torch.sum(output_real > 0.5).item()
        correct_fake = torch.sum(output_fake < 0.5).item()
        accuracy = (correct_real + correct_fake) / (2 * current_batch_size)
        accuracy_d.append(accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss_G: {avg_loss_g[-1]:.4f}, Loss_D: {avg_loss_d[-1]:.4f}, Accuracy_D: {accuracy:.4f}")

        # Atualizar o timer para a época atual
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        epoch_start_time = epoch_end_time
        total_training_time += epoch_time

        # Calculate Inception Score
        if inception_model is not None and (epoch % 5 == 0 or epoch == num_epochs - 1):  # Calculate every 5 epochs
            print("Calculando Inception Score...")
            with torch.no_grad():
                eval_samples = generator(torch.randn(min(500, batch_size*10), latent_dim, 1, 1).to(device))
                is_score, is_std = calculate_inception_score(eval_samples, inception_model)
                inception_scores.append(is_score)
                print(f'Epoch [{epoch+1}/{num_epochs}] - Inception Score: {is_score:.4f} ± {is_std:.4f}')

except Exception as e:
    print(f"Erro durante o treinamento: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Calcular a média dos Inception Scores ao final do treinamento
    if inception_scores:  # Verifica se a lista não está vazia
        average_inception_score = sum(inception_scores) / len(inception_scores)
        print(f'Média do Inception Score ao final do treinamento: {average_inception_score:.4f}')
    else:
        print('Nenhum Inception Score foi calculado.')


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
      f'Média Final Loss G: {avg_loss_g[-1]:.4f}\n'
      f'Média Final Loss D: {avg_loss_d[-1]:.4f}\n'
      f'Média Final Acc Real: {accuracy_d[-1]:.4f}\n'
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
