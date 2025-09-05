from ultralytics import YOLO
import cv2
import numpy as np
import os
import pandas as pd


model = YOLO("models/best_01_09_2025_monika_agatka_wiktoria_e_100_batch_8.pt")

def get_searching_results(img_path, confidence_level=0.5, save_path=None, result_preview_form="id"):
    results = model(img_path, conf=confidence_level)

    if save_path is not None:
        if result_preview_form == "id":
            img = cv2.imread(img_path)

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box[:4])
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Write element ID
                    cv2.putText(img, f"ID:{idx+1}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(save_path, img)
        else:
            for result in results:
                result.save(filename=save_path)

    return results

# Bacteria detecting algs

def get_tresh_mask_for_img(original_image_path,fixed=None):
    try:
        # Load the original image
        original_img = cv2.imread(original_image_path)
        if original_img is None:
            print(f"Error: Could not read the image '{original_image_path}'.")
            return []

    except Exception as e:
        print(f"An error occurred while loading the image: {e}")
        return []

    # Convert the image to grayscale and create the mask
    gray_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    if fixed:
        threshold_value, mask = cv2.threshold(
            gray_image, fixed, 255, cv2.THRESH_BINARY
        )
    else:
        threshold_value, mask = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Create a colored version of the mask (red)
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_colored[:, :, 0] = 0  # Zero out blue
    mask_colored[:, :, 1] = 0  # Zero out green

    # Overlay the mask on the original image
    overlay = cv2.addWeighted(original_img, 0.7, mask_colored, 0.3, 0)

    return threshold_value, mask, overlay

def contour_fit_percentage(contour):
    # Fit ellipse to contour
    ellipse = cv2.fitEllipse(contour)
    (center, axes, angle) = ellipse
    
    # Ellipse parameters
    cx, cy = center
    a = axes[0] / 2.0   # semi-major axis
    b = axes[1] / 2.0   # semi-minor axis
    theta = np.radians(angle)

    inside = 0
    total = len(contour)

    # Rotation matrix for ellipse alignment
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for pt in contour:
        x, y = pt[0]

        # Translate point relative to center
        xt = x - cx
        yt = y - cy

        # Rotate into ellipse-aligned coords
        xr = xt * cos_t + yt * sin_t
        yr = -xt * sin_t + yt * cos_t

        # Check ellipse equation
        if (xr**2) / (a**2) + (yr**2) / (b**2) <= 1:
            inside += 1

    return inside / total

def contour_thickness_along_minor_axis(contour, draw=True):
    # Fit ellipse
    ellipse = cv2.fitEllipse(contour)
    (center, axes, angle) = ellipse

    # Rotate contour so major axis is horizontal
    M = cv2.getRotationMatrix2D(center, angle + 90, 1.0)
    rotated = cv2.transform(contour, M)

    # Bounding box in rotated coordinates
    x, y, w, h = cv2.boundingRect(rotated)

    # Rasterize filled contour
    mask = np.zeros((y + h + 5, x + w + 5), dtype=np.uint8)
    shifted = rotated.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(mask, [shifted], -1, 255, cv2.FILLED)

    thicknesses = []
    lines = []

    for xi in range(mask.shape[1]):
        ys = np.where(mask[:, xi] > 0)[0]
        if len(ys) > 0:
            y_min, y_max = ys.min(), ys.max()
            thickness = y_max - y_min
            thicknesses.append(thickness)
            if draw:
                lines.append(((xi + x, int(y_min + y)), (xi + x, int(y_max + y))))

    # fallback: always have at least one value
    # if len(thicknesses) == 0:
    ys = rotated[:, 0, 1]
    # thicknesses.append(ys.max() - ys.min())
    thicknesses.append(min(axes))

    mean_thickness_px = np.mean(thicknesses)
    max_thickness_px = np.max(thicknesses)
    min_thickness_px = np.min(thicknesses)

    result = {
        "mean_thickness_px": mean_thickness_px,
        "max_thickness_px": max_thickness_px,
        "min_thickness_px": min_thickness_px
    }

    if draw:
        canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for p1, p2 in lines:
            cv2.line(canvas, p1, p2, (0, 0, 255), 1)
        result["canvas"] = canvas

    return result

def contour_thickness_along_major_axis(contour, draw=True):
    # Fit ellipse
    ellipse = cv2.fitEllipse(contour)
    (center, axes, angle) = ellipse
    cx, cy = center

    # Rotate contour so major axis is horizontal
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.transform(contour, M)

    # Bounding box in rotated coordinates
    x, y, w, h = cv2.boundingRect(rotated)

    thicknesses = []
    lines = []

    ys = rotated[:, 0, 1]
    y_min, y_max = ys.min(), ys.max()
    max_y_dim = y_max - y_min

    xs = np.round(rotated[:, 0, 0]).astype(int)
    unique_xs = np.unique(xs)


    # Rasterize filled contour
    mask = np.zeros((y + h + 5, x + w + 5), dtype=np.uint8)
    shifted = rotated.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(mask, [shifted], -1, 255, cv2.FILLED)

    thicknesses = []
    lines = []

    for xi in range(mask.shape[1]):
        ys = np.where(mask[:, xi] > 0)[0]
        if len(ys) > 0:
            y_min, y_max = ys.min(), ys.max()
            thickness = y_max - y_min
            thicknesses.append(thickness)
            if draw:
                lines.append(((xi + x, int(y_min + y)), (xi + x, int(y_max + y))))
                
    thicknesses.append(max(axes))
    mean_thickness_px = np.mean(thicknesses)
    max_thickness_px = np.max(thicknesses)
    min_thickness_px = np.min(thicknesses)

    result = {
        "mean_lenght_px": mean_thickness_px,
        "max_lenght_px":  max_thickness_px,
        "min_lenght_px":  min_thickness_px
    }

    if draw:
        canvas = np.zeros((y + h + 20, x + w + 20, 3), dtype=np.uint8)

        # Draw the filled contour first (with a light gray color)
        cv2.drawContours(canvas, [rotated], -1, (100, 100, 100), cv2.FILLED)
        cv2.drawContours(canvas, [rotated], -1, (0, 255, 0), 1)
        for p1, p2 in lines:  # draw subset of lines
            cv2.line(canvas, p1, p2, (0, 0, 255), 1)
        result["canvas"] = canvas

    return result

def analyze_bacterium_from_image(original_image_path, scale_factor_mm_per_pixel, total_image_area_mm2,x_pos=0,y_pos=0,bacteria_index=0,common_tresh=None):
    """
    Analyzes a single cropped image of a bacterium by:
    1. Creating a binary mask using Otsu's thresholding.
    2. Finding contours of the bacterium from the mask.
    3. Calculating all requested metrics in both pixels and millimeters.

    Args:
        original_image_path (str): Path to the cropped image file.
        scale_factor_mm_per_pixel (float): Conversion factor from pixels to millimeters.
        total_image_area_mm2 (float): The total area of the original image in mm^2.

    Returns:
        list of dict: A list of dictionaries with all calculated metrics.
    """
    try:
        # Load the original image
        original_img = cv2.imread(original_image_path)
        if original_img is None:
            print(f"Error: Could not read the image '{original_image_path}'.")
            return []

    except Exception as e:
        print(f"An error occurred while loading the image: {e}")
        return []

    # Convert the image to grayscale and create the mask
    gray_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)


    if common_tresh:
        try:
            threshold_value, mask = cv2.threshold(gray_image, common_tresh, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
            results = []
            object_count = len(contours)
            total_masked_area_px = cv2.countNonZero(mask)

            largest_contour = max(
                [c for c in contours if cv2.contourArea(c) >= 0],
                key=cv2.contourArea
            )
        except:
            return None # nothing was found - optionaly we can try use local masking #TODO
            # threshold_value, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    else:
        threshold_value, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print(f"Tresh: {threshold_value}")

    # gray_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # # First, compute Otsu threshold
    # otsu_thresh, _ = cv2.threshold(
    #     gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # )

    # # Apply again with adjusted threshold (otsu - 4)
    # _, mask = cv2.threshold(gray_image, otsu_thresh - 5, 255, cv2.THRESH_BINARY)


    
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # mask = cv2.dilate(mask, kernel, iterations=1)



    # Find contours in the mask (each contour represents a detected object)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    object_count = len(contours)
    total_masked_area_px = cv2.countNonZero(mask)

    largest_contour = max(
        [c for c in contours if cv2.contourArea(c) >= 0],
        key=cv2.contourArea
    )
    contour = largest_contour
        
    M = cv2.moments(contour)

    # 1. Pixel Coordinates
    x_pix = x_pos
    y_pix = y_pos
    
    # 2. Real-World Coordinates (mm)
    x_mm = x_pix * scale_factor_mm_per_pixel
    y_mm = y_pix * scale_factor_mm_per_pixel

    # 3. Area, Perimeter & Ellipse Metrics
    # contour_area_px = cv2.contourArea(contour)

    mask_2 = np.zeros_like(gray_image, dtype=np.uint8)
    cv2.drawContours(mask_2, [contour], -1, 255, cv2.FILLED)  # fill interior
    cv2.drawContours(mask_2, [contour], -1, 255, 1)           # add border

    # Now calculate area in pixels
    contour_area_px = np.count_nonzero(mask_2)

    contour_area_mm2 = contour_area_px * (scale_factor_mm_per_pixel ** 2)
    
    perimeter_px = cv2.arcLength(contour, True)
    perimeter_mm = perimeter_px * scale_factor_mm_per_pixel

    # Fit an ellipse to the contour
    try:
        contour_fit_in_ellipse = contour_fit_percentage(contour=contour)
        print(f"Countor in elipsa fit: {contour_fit_percentage(contour=contour)}")
        ellipse = cv2.fitEllipse(contour)
        (center, axes, angle) = ellipse
        # length_mm = max(axes) * scale_factor_mm_per_pixel

        contour_thickness_along_minor_axis_params = contour_thickness_along_minor_axis(contour=contour)
        contour_thickness_along_major_axis_params = contour_thickness_along_major_axis(contour=contour)

        # cv2.imshow("Contour Thickness", contour_thickness_along_major_axis_params["canvas"])
        # cv2.waitKey(0)

        # cv2.imshow("Contour Thickness", contour_thickness_along_minor_axis_params["canvas"])
        # cv2.waitKey(0)
        length_mm = contour_thickness_along_major_axis_params['max_lenght_px'] * scale_factor_mm_per_pixel

        # width_mm = min(axes) * scale_factor_mm_per_pixel * np.sqrt(np.sqrt(contour_fit_in_ellipse))
        width_mm = contour_thickness_along_minor_axis_params['max_thickness_px'] * scale_factor_mm_per_pixel
        
        # Perimeter of the fitted ellipse (Ramanujan's approximation)
        h = ((length_mm - width_mm) / (length_mm + width_mm)) ** 2
        perimeter_ellipse_mm = np.pi * (length_mm + width_mm) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
    except:
        # Fallback for contours that are too small or not ellipse-like
        length_mm, width_mm, angle = 0, 0, 0
        perimeter_ellipse_mm = perimeter_mm

    # 4. Feret's and Sieve Diameter
    rect = cv2.minAreaRect(contour)
    width_rect_px = rect[1][0]
    height_rect_px = rect[1][1]
    
    feret_h_mm = max(width_rect_px, height_rect_px) * scale_factor_mm_per_pixel
    feret_v_mm = min(width_rect_px, height_rect_px) * scale_factor_mm_per_pixel
    
    # Martin's Diameter (using your simplification: major axis of the fitted ellipse)
    martin_diameter_mm = length_mm

    # Sieve Diameter: diameter of a circle with the same area
    sieve_diameter_mm = 2 * np.sqrt(contour_area_mm2 / np.pi)
    
    # 5. Color and Brightness Metrics
    object_mask = np.zeros(original_img.shape[:2], dtype="uint8")
    cv2.drawContours(object_mask, [contour], -1, 255, -1)
    
    object_pixels = original_img[object_mask == 255]
    
    b = object_pixels[:, 0]  # All rows, first column (blue)
    g = object_pixels[:, 1]  # All rows, second column (green)
    r = object_pixels[:, 2]  # All rows, third column (red)
    mean_b = np.mean(b) if b.size > 0 else 0
    mean_g = np.mean(g) if g.size > 0 else 0
    mean_r = np.mean(r) if r.size > 0 else 0
    
    mean_y = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b
    
    gray_original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Now, use the mask to extract only the pixels belonging to the bacterium from the grayscale image
    object_pixels_gray = gray_original_img[object_mask == 255]

    # Calculate the mean brightness of those extracted pixels
    mean_brightness = np.mean(object_pixels_gray) if object_pixels_gray.size > 0 else 0
    
    # 6. Total Image Metrics
    udzial_punktow = (total_masked_area_px * (scale_factor_mm_per_pixel ** 2) / total_image_area_mm2) * 100 if total_image_area_mm2 > 0 else 0

    results.append({
        'nr.': bacteria_index,
        'xpix.': x_pix,
        'ypix.': y_pix,
        'xmm': round(x_mm,1),
        'ymm': round(y_mm,1),
        'powierzchniamm': round(contour_area_mm2,2),
        'dlugoscmm': round(length_mm,8),
        'szerokoscmm': round(width_mm,8),
        'kat': round(angle,2),
        'obwodmm': round(perimeter_mm,8),
        'obwod_c.mm': round(perimeter_ellipse_mm,8),
        'srednica_fereta_hmm': round(feret_h_mm,1),
        'srednica_fereta_vmm': round(feret_v_mm,1),
        'sredn._martinamm': round(martin_diameter_mm,1),
        'sredn._sitowamm': round(sieve_diameter_mm,8),
        'srednia_jaskrawosc': round(mean_brightness,1),
        'r': round(mean_r,1),
        'g': round(mean_g,1),
        'b': round(mean_b,1),
        'y': round(mean_y,1),
        'liczenie_obiektow_w': int(object_count),
        'udzial_punktow': round(udzial_punktow,1),
        'pole_obrazu_mm2': round(total_image_area_mm2,1)
    })
        
    return results

def get_results_df(full_image_path,results,common_tresh=None):
    image = cv2.imread(full_image_path)

    results_list = []

    if len(results[0]) == 0:
            results_list = {
                'nr.': [],
                'xpix.': [],
                'ypix.': [],
                'xmm': [],
                'ymm': [],
                'powierzchniamm': [],
                'dlugoscmm': [],
                'szerokoscmm': [],
                'kat': [],
                'obwodmm': [],
                'obwod_c.mm': [],
                'srednica_fereta_hmm': [],
                'srednica_fereta_vmm': [],
                'sredn._martinamm': [],
                'sredn._sitowamm': [],
                'srednia_jaskrawosc': [],
                'r': [],
                'g': [],
                'b': [],
                'y': [],
                'liczenie_obiektow_w': [],
                'udzial_punktow': [],
                'pole_obrazu_mm2': []
            }


    common_tresh,commmon_mask,overlay = get_tresh_mask_for_img(full_image_path,fixed=common_tresh)

    # Ensure results[0] contains bounding boxes in format [x1, y1, x2, y2]
    for i, box in enumerate(results[0]):
        for det in box.boxes:
            x1, y1, x2, y2 = det.xyxy.int().tolist()[0] # Convert coordinates to integers

            # Crop the region of interest
            cropped = image[y1:y2, x1:x2]

            # Save cropped image to temporary path
            cropped_path = f"temp_cropped_{i}.jpg"
            cv2.imwrite(cropped_path, cropped)

            # Analysis parameters
            scale_mm_per_px = 0.1
            total_image_area_mm2 = abs(x1-x2)*abs(y1-y2)*scale_mm_per_px*scale_mm_per_px

            # Run analysis on the cropped image
            analysis_results = analyze_bacterium_from_image(cropped_path, scale_mm_per_px, total_image_area_mm2,x_pos=(x1+x2)/2,y_pos=image.shape[0]-(y1+y2)/2,bacteria_index=i+1,common_tresh=common_tresh)


            if analysis_results and analysis_results != None:
                for result in analysis_results:
                    results_list.append(result)
            else:
                print(f"Analysis failed or no objects were detected for Box #{i}.")

            # Optionally remove the temporary cropped image
            os.remove(cropped_path)

    return pd.DataFrame(results_list)


# Macro replacement

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df != "******").all(axis=1)]
    
    col_map = {
        "długośćmm": "dlugoscmm",
        "szerokośćmm": "szerokoscmm",
        "kąt": "kat",
        "obwódmm": "obwodmm",
        "średnica_fereta_hmm": "srednica_fereta_hmm",
        "średn._martinamm": "sredn._martinamm",
        "średn._sitowamm": "sredn._sitowamm",
        "średnia_jaskrawość": "srednia_jaskrawosc"
    }

    if "długośćmm" in df.columns:
        df = df.rename(columns=col_map)
        if "szerokoscmm" in df.columns:
            df["szerokoscmm"] = df["szerokoscmm"].astype(np.float64)

    return df

def get_individual_stats(df,is_pred=False):
    df = preprocess_df(df)
    # Scaling
    df['dlugoscmm'] = df['dlugoscmm']/1.49
    df['szerokoscmm'] = df['szerokoscmm']/1.49
    df['powierzchniamm'] = df['powierzchniamm']/2.235

    # filtering the things that with high probability are not bacteria !TODO odkomentuj to po testach
    # df = df[df['dlugoscmm'] >= 0.2]
    # df = df[df['szerokoscmm'] <= 1.5]

    df['Pw'] = 3.14*(df['szerokoscmm']/2)**2+df['szerokoscmm']*(df['dlugoscmm']-df['szerokoscmm'])
    df['R'] = df['Pw'] / df['powierzchniamm']
    df['Dk'] = df['dlugoscmm']
    df['Sk'] = df['szerokoscmm']
    df['D/S'] = (df['Dk'] / df['Sk']).astype('float64')

    bacteria_types = []
    Dks = []
    Sks = []
    for idx,row  in df.iterrows():
        # if is_pred:
        #     row["R"] -= 0.5 
        #     row['D/S'] += 0.2
        if row['dlugoscmm'] < 0.2 or row['szerokoscmm'] > 1.5:
            Dks.append(row['dlugoscmm'])
            Sks.append(row['szerokoscmm'])
            bacteria_types.append("Removed_dim")
        elif row['R'] > 1.2:
            Dks.append(row['szerokoscmm'])
            Sks.append(row['Pw']*0.8)
            bacteria_types.append("Krzywe")
        else:
            Dks.append(row['dlugoscmm'])
            Sks.append(row['szerokoscmm'])
            if round(row['D/S'],15) > 1.5:
                bacteria_types.append("Pałeczki")
            else:
                bacteria_types.append("Ziarniaki")
    df['bacteria_type'] = bacteria_types
    df['Dk'] = Dks
    df['Sk'] = Sks

    df['Ob'] =(3.14*(df['Sk']**3)/6)+(3.14*((df['Sk']**2)/4)*(df['Dk']-df['Sk']))
    # !TODO remove this
    # df = df[df['Ob']>0]
    df.loc[df['Ob'] <= 0, 'bacteria_type'] = "removed_ob"
    # df['Ob'] = np.abs(df['Ob'])
    # -----------------

    df['bialko'] = 104.5 * (df['Ob']**0.59)
    df['wegiel'] = 0.86 * df['bialko']

    return df

def get_stats_for_bacteria_types(df,probe_volume_ml = 6):
    grouped = df.groupby("bacteria_type")
    result_count = grouped.size().reset_index(name="count")
    result = grouped["Ob"].mean().reset_index()

    result['bialko'] = 104.5 * (result['Ob']**0.59)
    result['wegiel'] = 0.86 * result['bialko']
    result['count_in_1_ml']=((result_count['count']*48097.39)/10)/(probe_volume_ml)
    result['biomasa'] =(((104.5*result['Ob']**0.59)*0.86)*result['count_in_1_ml'])/1000000

    return result

import numpy as np

def get_stats_for_for_ob_bins(df,probe_volume_ml = 6):
    bins = [0, 0.1, 0.2, 0.5, 1.0, float("inf")]
    labels = ["<=0.1", "0.1–0.2", "0.2–0.5", "0.5–1.0", ">1.0"]

    df["Ob_bucket"] = pd.cut(df["Ob"], bins=bins, labels=labels, right=True)

    grouped = df.groupby(["bacteria_type", "Ob_bucket"], observed=True)
    result_bio_stats = grouped.size().reset_index(name="count")

    total_bacteria_count = np.sum(result_bio_stats['count'])
    total_bacteria_count_1_ml = ((np.sum(result_bio_stats['count'])*48097.39)/10)/(probe_volume_ml)

    result_bio_stats['count_in_1_ml']=((result_bio_stats['count']*48097.39)/10)/(probe_volume_ml)
    result_bio_stats['bio_diversity'] =((result_bio_stats['count_in_1_ml']+1)/total_bacteria_count_1_ml)*np.log10((result_bio_stats['count_in_1_ml']+1)/total_bacteria_count_1_ml)

    return result_bio_stats

def get_speified_baceria_types_count(df):
    return df.groupby("bacteria_type").size().reset_index(name="count")

def get_shannon_index(result_bio_stats):
    return np.sum(result_bio_stats['bio_diversity'])*-1

def full_analyse(df,proube_volume_ml=6,is_pred = False):
    if len(df) == 0:
        return pd.DataFrame(columns=["bacteria_type","count"]),pd.DataFrame()

    df = preprocess_df(df)
    df = get_individual_stats(df,is_pred=is_pred)
    result = get_stats_for_bacteria_types(df,probe_volume_ml=proube_volume_ml)
    result_bio_stats = get_stats_for_for_ob_bins(df,probe_volume_ml=proube_volume_ml)
    specified_types_count_predicted =  get_speified_baceria_types_count(df)
    shannon_index = np.sum(result_bio_stats['bio_diversity'])*-1
    print(f"Shannon index: {shannon_index}")

    return specified_types_count_predicted,df