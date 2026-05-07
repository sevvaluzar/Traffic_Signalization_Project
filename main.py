import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

class IntersectionEnv:
    def __init__(self, max_state_queue=5):
        # 4 yön: Kuzey(0), Güney(1), Doğu(2), Batı(3)
        self.queues = [0, 0, 0, 0]
        self.max_state_queue = max_state_queue
        
        # Simülasyon parametreleri
        self.arrival_rates = [0.4, 0.4, 0.3, 0.3]  # Her adımda araç gelme olasılığı (Poisson/Binomial)
        self.departure_rate = 2  # Yeşil yandığında bir adımda geçebilecek maksimum araç sayısı

        self.current_step = 0
        self.max_steps = 100 # Bir bölümdeki maksimum adım sayısı

    def reset(self):
        self.queues = [0, 0, 0, 0]
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        # Q-Learning için durum uzayını sınırla (discrete state space)
        # Örn: Eğer kuyruk 5'ten büyükse 5 olarak kabul et.
        state = tuple(min(q, self.max_state_queue) for q in self.queues)
        return state

    def step(self, action):
        """
        action 0: Kuzey Yeşil
        action 1: Güney Yeşil
        action 2: Doğu Yeşil
        action 3: Batı Yeşil
        """
        # 1. Araçların ayrılması (Sadece yeşil olan yönlerde)
        if 0 <= action <= 3:
            self.queues[action] = max(0, self.queues[action] - self.departure_rate)

        # 2. Yeni araçların gelmesi
        for i in range(4):
            # Basit bir rastgele varış modeli
            arrivals = np.random.poisson(self.arrival_rates[i])
            self.queues[i] += arrivals

        self.current_step += 1
        
        # 3. Ödül hesaplama (Negatif toplam kuyruk uzunluğu)
        # Amacımız kuyrukların toplamını minimize etmek
        total_waiting = sum(self.queues)
        reward = -total_waiting

        # 4. Bitiş kontrolü
        done = self.current_step >= self.max_steps

        return self._get_state(), reward, done, {"total_waiting": total_waiting}


class QLearningAgent:
    def __init__(self, action_size=4, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.action_size = action_size

    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        return self.q_table[state][action]

    def choose_action(self, state, train=True):
        if train and random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Keşif (Exploration)
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state])  # Sömürü (Exploitation)

    def learn(self, state, action, reward, next_state, done):
        predict = self.get_q_value(state, action)
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size)

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] += self.lr * (target - predict)

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def train_agent(episodes=500):
    env = IntersectionEnv()
    agent = QLearningAgent()
    rewards_history = []
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)
        if (e+1) % 100 == 0:
            print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward} - Epsilon: {agent.epsilon:.3f}")
            
    return agent, rewards_history


def evaluate_baselines(agent, test_episodes=50):
    env = IntersectionEnv()
    
    rl_wait_times = []
    fixed_wait_times = []
    
    for _ in range(test_episodes):
        # RL Ajanını Test Et
        state = env.reset()
        done = False
        total_wait_rl = 0
        while not done:
            action = agent.choose_action(state, train=False)
            state, _, done, info = env.step(action)
            total_wait_rl += info["total_waiting"]
        rl_wait_times.append(total_wait_rl)

        # Sabit Zamanlı Sistemi Test Et (Örn: Her 5 adımda bir ışığı değiştir)
        env.reset()
        done = False
        total_wait_fixed = 0
        action = 0
        step_counter = 0
        while not done:
            if step_counter % 5 == 0:
                action = (action + 1) % 4  # Sırayla 0, 1, 2, 3
            _, _, done, info = env.step(action)
            total_wait_fixed += info["total_waiting"]
            step_counter += 1
        fixed_wait_times.append(total_wait_fixed)

    return rl_wait_times, fixed_wait_times


def plot_results(rewards, rl_waits, fixed_waits):
    plt.figure(figsize=(12, 5))

    # Eğitim Ödülleri Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(rewards, color='blue', alpha=0.6)
    # Hareketli ortalama ekleyelim
    window = 20
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, color='red', linewidth=2, label='Moving Avg')
    plt.title("Eğitim Sürecinde Toplam Ödül")
    plt.xlabel("Bölüm (Episode)")
    plt.ylabel("Ödül (Negatif Bekleme Süresi)")
    plt.legend()

    # Karşılaştırma Grafiği
    plt.subplot(1, 2, 2)
    plt.boxplot([rl_waits, fixed_waits], tick_labels=['RL Adaptif Sistem', 'Sabit Zamanlı Sistem'])
    plt.title("Test Aşaması: Toplam Bekleme Süreleri")
    plt.ylabel("Kümülatif Bekleyen Araç Sayısı")

    plt.tight_layout()
    plt.savefig("sonuclar.png", dpi=300)
    print("Grafik 'sonuclar.png' olarak kaydedildi.")
    # plt.show() # Konsol arayüzünde show durduracağı için sadece kaydetmek daha iyi.

def generate_gif(agent, filename="simulation.gif", steps=50):
    env = IntersectionEnv()
    state = env.reset()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    directions = ['Kuzey', 'Güney', 'Doğu', 'Batı']
    
    def update(frame):
        nonlocal state
        ax.clear()
        
        action = agent.choose_action(state, train=False)
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        queues = env.queues
        colors = ['red'] * 4
        colors[action] = 'green'
                  
        bars = ax.bar(directions, queues, color=colors)
        
        ax.set_ylim(0, max(20, max(queues) + 5))
        ax.set_ylabel('Kuyruk Uzunluğu (Araç Sayısı)')
        
        green_dir = directions[action]
        ax.set_title(f"Adım: {frame} | Yeşil Işık: {green_dir}\nToplam Bekleyen: {info['total_waiting']}")
        
        return bars

    print(f"{filename} animasyonu oluşturuluyor...")
    ani = animation.FuncAnimation(fig, update, frames=steps, interval=1000, blit=False)
    
    writer = animation.PillowWriter(fps=1)
    ani.save(filename, writer=writer)
    print(f"Animasyon '{filename}' olarak kaydedildi.")
    plt.close(fig)

def generate_grid_gif(agent, filename="grid_simulation.gif", steps=50):
    env = IntersectionEnv()
    state = env.reset()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(frame):
        nonlocal state
        ax.clear()
        
        action = agent.choose_action(state, train=False)
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        # Yol çizgilerini çiz
        ax.plot([-5, -1], [1, 1], 'k-', lw=2)
        ax.plot([-5, -1], [-1, -1], 'k-', lw=2)
        ax.plot([1, 5], [1, 1], 'k-', lw=2)
        ax.plot([1, 5], [-1, -1], 'k-', lw=2)
        
        ax.plot([-1, -1], [1, 5], 'k-', lw=2)
        ax.plot([1, 1], [1, 5], 'k-', lw=2)
        ax.plot([-1, -1], [-5, -1], 'k-', lw=2)
        ax.plot([1, 1], [-5, -1], 'k-', lw=2)
        
        # Şerit kesik çizgileri
        ax.plot([-5, -1], [0, 0], 'k--', lw=1)
        ax.plot([1, 5], [0, 0], 'k--', lw=1)
        ax.plot([0, 0], [1, 5], 'k--', lw=1)
        ax.plot([0, 0], [-5, -1], 'k--', lw=1)
        
        # Işık Renkleri
        n_color = 'green' if action == 0 else 'red'
        s_color = 'green' if action == 1 else 'red'
        e_color = 'green' if action == 2 else 'red'
        w_color = 'green' if action == 3 else 'red'
        
        # Trafik ışıklarını (daire) çiz
        ax.plot(-0.5, 1.2, marker='o', color=n_color, markersize=10) # Kuzeyden gelene
        ax.plot(0.5, -1.2, marker='o', color=s_color, markersize=10) # Güneyden gelene
        ax.plot(1.2, 0.5, marker='o', color=e_color, markersize=10) # Doğudan gelene
        ax.plot(-1.2, -0.5, marker='o', color=w_color, markersize=10) # Batıdan gelene
        
        # Araçları (noktalar) çiz
        queues = env.queues # [Kuzey, Güney, Doğu, Batı]
        
        # Kuzeyden gelen araçlar (Aşağı iniyor) x = -0.5
        for i in range(queues[0]):
            ax.plot(-0.5, 1.5 + i*0.5, marker='s', color='blue', markersize=8)
            
        # Güneyden gelen araçlar (Yukarı çıkıyor) x = 0.5
        for i in range(queues[1]):
            ax.plot(0.5, -1.5 - i*0.5, marker='s', color='orange', markersize=8)
            
        # Doğudan gelen araçlar (Sola gidiyor) y = 0.5
        for i in range(queues[2]):
            ax.plot(1.5 + i*0.5, 0.5, marker='s', color='purple', markersize=8)
            
        # Batıdan gelen araçlar (Sağa gidiyor) y = -0.5
        for i in range(queues[3]):
            ax.plot(-1.5 - i*0.5, -0.5, marker='s', color='cyan', markersize=8)
            
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.axis('off') # Eksenleri gizle
        
        green_dir = ['Kuzey', 'Güney', 'Doğu', 'Batı'][action]
        ax.set_title(f"Kavşak Görünümü - Adım: {frame}\nYeşil Işık: {green_dir} | Toplam Bekleyen: {info['total_waiting']}")
        
    print(f"{filename} animasyonu oluşturuluyor...")
    ani = animation.FuncAnimation(fig, update, frames=steps, interval=1000, blit=False)
    
    writer = animation.PillowWriter(fps=1)
    ani.save(filename, writer=writer)
    print(f"Animasyon '{filename}' olarak kaydedildi.")
    plt.close(fig)

if __name__ == "__main__":
    print("RL Ajanı eğitiliyor...")
    trained_agent, training_rewards = train_agent(episodes=1000)
    
    print("\nTest ve karşılaştırma yapılıyor...")
    rl_wait_results, fixed_wait_results = evaluate_baselines(trained_agent, test_episodes=100)
    
    avg_rl = np.mean(rl_wait_results)
    avg_fixed = np.mean(fixed_wait_results)
    
    print(f"\nOrtalama Toplam Bekleme (RL Adaptif): {avg_rl:.2f}")
    print(f"Ortalama Toplam Bekleme (Sabit Zamanlı): {avg_fixed:.2f}")
    print(f"İyileşme Oranı: % {((avg_fixed - avg_rl) / avg_fixed) * 100:.2f}")
    
    plot_results(training_rewards, rl_wait_results, fixed_wait_results)
    
    generate_gif(trained_agent, filename="simulation.gif", steps=50)
    generate_grid_gif(trained_agent, filename="grid_simulation.gif", steps=50)
