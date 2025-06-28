import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
from get_solar_data import get_solar_power

# Load roof and obstruction masks
main_dir = 'test_crowd/original'
roof_dir = f'test_crowd/masks'

for file in os.listdir(roof_dir):

    total_pon = 0
    PIXEL_RESOLUTION = 0.01
    PEAK_POWER_PER_PANEL = 0.32 # 0.32kWp 
    SOLAR_POWER = get_solar_power()

    roof_path = os.path.join(roof_dir, file)
    roof_mask = np.load(roof_path)      # binary [0,1], shape (512,512)

    # Label each roof (connected components)
    num_labels, roof_labels, stats, centroids = cv2.connectedComponentsWithStats(roof_mask.astype(np.uint8), connectivity=8)

    # print(centroids.shape)
    # print(centroids)
    # For each roof, calculate area info
    roof_data = []
    colored_roof_img = np.zeros((roof_mask.shape[0], roof_mask.shape[1], 3), dtype=np.uint8)
    np.random.seed(42)
    # print(colored_roof_img.shape)

    roof_mask = np.zeros((roof_mask.shape[0], roof_mask.shape[1]),dtype=float)
    # print(roof_mask.shape)

    for roof_id in range(1, num_labels):  # Skip label 0 (background)
        roof_area = stats[roof_id, cv2.CC_STAT_AREA]
        if(roof_area > 75):
            mask = (roof_labels == roof_id)
            roof_mask[mask] = roof_id

            # Get obstructions within this roof
            net_area = roof_area

            actual_area = net_area * (PIXEL_RESOLUTION ** 2)
            panels = int(actual_area / 2)
            peak_power = panels * PEAK_POWER_PER_PANEL
            solar_potential = peak_power * SOLAR_POWER
            total_pon += solar_potential

            # Assign random color for visualization
            color = np.random.randint(50, 255, size=3)
            colored_roof_img[mask] = color

            # Store results
            roof_data.append({
                'Roof_ID': roof_id,
                'Roof_Area': roof_area,
                'Net_Usable_Area': net_area,
                'Real_Area': round(actual_area, 3),
                'Panels': panels,
                'Solar_potential_per_year': round(solar_potential, 3)
            })
    
    with open(f'{main_dir}/total_pon/{os.path.splitext(file)[0]}.txt', 'a') as f:
        f.write(str(total_pon))


    # Save CSV
    np.save(f'{main_dir}/com_bin/{os.path.splitext(file)[0]}.npy', roof_mask)

    df = pd.DataFrame(roof_data)
    df.to_csv(f'{main_dir}/roof_data/{os.path.splitext(file)[0]}.csv', index=False)
    print("Saved to roof_area_data.csv")

    # Save visualization image with labels
    for item in roof_data:
        cx, cy = centroids[item['Roof_ID']]
        plt.text(int(cx), int(cy), str(item['Roof_ID']), fontsize=8, color='white')

    plt.imshow(colored_roof_img)
    plt.axis('off')
    plt.title("Numbered Roofs")
    plt.savefig(f'{main_dir}/components/{os.path.splitext(file)[0]}.png', bbox_inches='tight', dpi=300)
    plt.close()
