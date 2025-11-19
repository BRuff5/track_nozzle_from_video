import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

video_path = "printline.MOV"    
template_path = "nozzle_template.jpg"
frame_skip = 2


# ---- LOAD VIDEO & TEMPLATE ----
cap = cv2.VideoCapture(video_path)
#I added a safety check for template loading
# this helps YOU know if nozzle_template.jpg is not found.
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
if template is None:
    raise FileNotFoundError(f"Could not load template at {template_path}")

# Make template smoother / more robust (add blur)
#helps matching when images are noisy or slightly different.
template = cv2.GaussianBlur(template, (3, 3), 0)

w, h = template.shape[::-1]
positions = []

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % frame_skip != 0:
        frame_idx += 1
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #  Match preprocessing to template
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # ---- OBJECT RECOGNITION USING TEMPLATE MATCHING ----
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    #  Use a slightly lower threshold to catch more matches
    #    (you can tune between 0.5–0.75 depending on your data)
    #This prevents the nozzle from being missed.
    if max_val > 0.55: 
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        center = (top_left[0] + w // 2, top_left[1] + h // 2)
        positions.append(center)

        # Visualization
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        cv2.putText(frame, f"{max_val:.2f}", (top_left[0], top_left[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    else:
        # Optional debug: see how strong the best match is
        # print(f"Max match value this frame: {max_val:.3f}")
        pass

    cv2.imshow("Object Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print(f"✅ Tracked {len(positions)} nozzle positions")

# ---- NORMALIZE & SCALE ----
positions = np.array(positions)
positions = positions - np.min(positions, axis=0)
positions = positions / np.max(positions, axis=0) * 100

# ---- PLOT PATH ----
plt.plot(positions[:, 0], positions[:, 1], 'r-')
plt.gca().invert_yaxis()
plt.title("Nozzle Path (Object Recognition Tracking)")
plt.show()

# ---- EXPORT GCODE ----
gcode_lines = [
    "; Auto-generated nozzle path via Object Detection",
    "G21 ; mm mode",
    "G90 ; absolute",
    "G1 F1200"
]
for x, y in positions:
    gcode_lines.append(f"G1 X{x:.2f} Y{y:.2f} E0.02")

with open("tracked_path.gcode", "w") as f:
    f.write("\n".join(gcode_lines))

print("\n--- GCODE PREVIEW ---")
print("\n".join(gcode_lines[:30]))
print("\n💾 Full G-code saved as tracked_path.gcode")
