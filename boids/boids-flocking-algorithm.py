# кожен дрон - це об'єкт, який має координати, швидкість, прискорення
# на кожному кроці розраховує три вектори силиЖ
# - розділення (separation) - відштовхування від сусідів
# - вирівнювання (alignment) - вирівнювання швидкості з сусідами
# - згуртування (cohesion) - притягування до центру маси сусідів
# потім оновлює швидкість та координати дрона

# Реалізація

# на кожній ітерації циклу (30 секунд) кожен дрон:
# 1. знаходить всіх сусідів в радіусі 100 метрів
# 2. розраховує три вектори сили
# 3. оновлює швидкість та координати дрона

# Python implementation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class BoidSimulation:
    def __init__(self, num_boids=50, width=100, height=100):
        self.num_boids = num_boids
        self.width = width
        self.height = height

        # 1. Ініціалізація позицій та швидкостей
        # Випадкові позиції (x, y)
        self.positions = np.random.rand(num_boids, 2) * np.array([width, height])
        
        # Випадкові швидкості (vx, vy), нормалізовані
        angles = np.random.rand(num_boids) * 2 * np.pi
        velocities = np.column_stack((np.cos(angles), np.sin(angles)))
        self.velocities = velocities * 3  # Початкова швидкість

        # 2. Параметри (The "Knobs" - налаштування поведінки)
        self.visual_range = 10.0  # Радіус бачення
        self.protected_range = 2.0 # Радіус особистого простору (щоб не врізатися)
        
        # Вага правил (Weights)
        self.separation_factor = 0.05
        self.alignment_factor = 0.05
        self.cohesion_factor = 0.01
        
        # Фізичні обмеження
        self.max_speed = 4.0
        self.min_speed = 2.0
        self.turn_factor = 0.2 # Інерція повороту

    def update(self):
        # Отримуємо сили для кожного агента
        separation = np.zeros((self.num_boids, 2))
        alignment = np.zeros((self.num_boids, 2))
        cohesion = np.zeros((self.num_boids, 2))

        # --- Векторизація не використовується для ясності логіки (O(N^2)) ---
        # Для оптимізації тут варто використовувати cKDTree або матриці відстаней NumPy
        
        for i in range(self.num_boids):
            neighbors_count = 0
            pos_avg = np.zeros(2)
            vel_avg = np.zeros(2)
            close_dx, close_dy = 0.0, 0.0

            for j in range(self.num_boids):
                if i == j: continue # Не порівнювати з собою

                # Вектор від i до j
                dx = self.positions[i][0] - self.positions[j][0]
                dy = self.positions[i][1] - self.positions[j][1]
                distance = np.sqrt(dx*dx + dy*dy)

                # 1. Separation (Розділення) - критична зона
                if distance < self.protected_range:
                    close_dx += dx
                    close_dy += dy

                # 2. Alignment & Cohesion (Вирівнювання та Згуртованість) - зона видимості
                if distance < self.visual_range:
                    pos_avg += self.positions[j]
                    vel_avg += self.velocities[j]
                    neighbors_count += 1

            # Застосування Separation
            separation[i] = np.array([close_dx, close_dy]) * self.separation_factor

            # Застосування Alignment & Cohesion (якщо є сусіди)
            if neighbors_count > 0:
                pos_avg /= neighbors_count
                vel_avg /= neighbors_count

                # Cohesion: вектор до центру мас сусідів
                cohesion_vector = pos_avg - self.positions[i]
                cohesion[i] = cohesion_vector * self.cohesion_factor

                # Alignment: різниця між середньою швидкістю і моєю
                alignment_vector = vel_avg - self.velocities[i]
                alignment[i] = alignment_vector * self.alignment_factor

        # Оновлення швидкості з урахуванням усіх сил
        self.velocities += separation + alignment + cohesion

        # Обмеження швидкості (Speed Limiter)
        speeds = np.linalg.norm(self.velocities, axis=1)
        # Уникаємо ділення на нуль
        mask = speeds > self.max_speed
        self.velocities[mask] = (self.velocities[mask] / speeds[mask][:, np.newaxis]) * self.max_speed
        
        # Мінімальна швидкість (щоб вони не зупинялися)
        mask_min = speeds < self.min_speed
        # Якщо швидкість 0, даємо випадковий поштовх, інакше нормалізуємо
        # Спрощено: просто масштабуємо, якщо не нуль
        non_zero = speeds > 0.01
        mask_update = mask_min & non_zero
        self.velocities[mask_update] = (self.velocities[mask_update] / speeds[mask_update][:, np.newaxis]) * self.min_speed

        # Оновлення позиції
        self.positions += self.velocities

        # Граничні умови (Wrap around / Toroidal space)
        # Якщо вилітає справа, з'являється зліва
        self.positions = self.positions % np.array([self.width, self.height])

# --- Візуалізація ---
sim = BoidSimulation(num_boids=70, width=100, height=100)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, sim.width)
ax.set_ylim(0, sim.height)
ax.set_title("Reynolds Boids Simulation (Python + Matplotlib)")

# Малюємо дронів як стрілочки (Quiver plot)
quiver = ax.quiver(
    sim.positions[:, 0], 
    sim.positions[:, 1], 
    sim.velocities[:, 0], 
    sim.velocities[:, 1],
    color='blue', 
    headwidth=3, 
    scale=40
)

def animate(frame):
    sim.update()
    
    # Оновлюємо дані на графіку
    quiver.set_offsets(sim.positions)
    quiver.set_UVC(sim.velocities[:, 0], sim.velocities[:, 1])
    return quiver,

anim = FuncAnimation(fig, animate, frames=200, interval=30, blit=True)

plt.show()