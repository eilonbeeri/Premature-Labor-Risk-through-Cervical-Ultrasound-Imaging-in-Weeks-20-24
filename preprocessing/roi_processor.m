% הגדרת נתיב התיקייה ממנה ניקח את התמונות
input_dir = 'C:\Users\eilon\Downloads\rotated\';

% טען את התמונה (החלון ייפתח ישירות בתיקייה שהוגדרה)
[file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp'}, 'Select ultrasound image', input_dir);

% אם המשתמש ביטל את בחירת התמונה, צא מהסקריפט
if isequal(file,0)
   disp('User selected Cancel');
   return;
end

image = imread(fullfile(path, file));

% הגדרת תיקיית השמירה הקבועה
output_folder = 'C:\Users\eilon\Downloads\ROI\';

% יצירת התיקייה במידה והיא עדיין לא קיימת
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% הגדל את התמונה פי 4
scale_factor = 4;
resized_image = imresize(image, scale_factor);

% הצג את התמונה המוגדלת לסימון קו כיול
figure;
imshow(resized_image);
title('1. עשה זום לאזור הרצוי בעזרת סרגל הכלים. 2. לחץ על מקש כלשהו כדי להתחיל לסמן');
zoom on; % מדליק את אפשרות הזום
pause;   % עוצר את הקוד ומחכה שתלחץ על מקש במקלדת
zoom off; % מכבה את הזום כדי לאפשר לחיצות

title('סמן כעת קו כיול באורך ידוע (2 נקודות)');

% Select 2 points to define the calibration line
[x_line, y_line] = ginput(2);

% Draw the selected line for visualization
hold on;
plot(x_line, y_line, 'r-', 'LineWidth', 2);

% Calculate the length of the drawn line in pixels
line_pos = [x_line y_line];
line_height = norm(line_pos(1,:) - line_pos(2,:));

% Ask user for the real length of the drawn line in cm
real_line_length_cm = input('Enter the real length of the line you drew in cm: ');

% Calculate rectangle height in pixels for 2 cm
rect_height_pixels = (line_height / real_line_length_cm) * 2;

% הצג את התמונה המוגדלת שוב לסימון הנקודות עבור המלבן
figure;
imshow(resized_image);
title('1. עשה זום לאזור המלבן. 2. לחץ על מקש כלשהו כדי להתחיל לסמן');
zoom on;
pause;
zoom off;

title('סמן כעת 4 נקודות עבור המלבן');

% סמן 4 נקודות בתמונה המוגדלת
points = ginput(4);

% חשב את המרכז של הנקודות
center_point = mean(points, 1);

% חישוב זווית כל נקודה יחסית למרכז וסידור הנקודות
angles = atan2(points(:,2) - center_point(2), points(:,1) - center_point(1));
[~, sorted_idx] = sort(angles);
sorted_points = points(sorted_idx, :);

% חשב את הרוחב על פי נקודות המלבן
width = norm(sorted_points(2,:) - sorted_points(1,:));

% מצא את זווית הסיבוב
theta = atan2(sorted_points(2,2) - sorted_points(1,2), sorted_points(2,1) - sorted_points(1,1));

% צור את המלבן המיושר
R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
rect_points = [0, 0; width, 0; width, rect_height_pixels; 0, rect_height_pixels];
aligned_points = (R * rect_points')' + sorted_points(1,:);

% ציור המלבן המיושר על התמונה המוגדלת
hold on;
plot([aligned_points(:,1); aligned_points(1,1)], [aligned_points(:,2); aligned_points(1,2)], 'r-', 'LineWidth', 2);
title('התמונה המקורית עם המלבן המיושר בגובה 2cm');

% חזור לגודל המקורי של התמונה לפני החיתוך
aligned_points = aligned_points / scale_factor;

% חתוך את התמונה המקורית לפי המלבן המיושר
xmin = min(aligned_points(:,1));
xmax = max(aligned_points(:,1));
ymin = min(aligned_points(:,2));
ymax = max(aligned_points(:,2));

% ודא שהמיקומים בתוך הגבולות של התמונה המקורית
xmin = max(1, xmin);
ymin = max(1, ymin);
xmax = min(size(image, 2), xmax);
ymax = min(size(image, 1), ymax);

cropped_image = imcrop(image, [xmin ymin xmax-xmin ymax-ymin]);

% הצג את התמונה שנחתכה
figure;
imshow(cropped_image);
title('התמונה שנחתכה לפי המלבן המיושר');

% שמור את התמונה החתוכה באותו שם מקורי לחלוטין
output_name = fullfile(output_folder, file);
imwrite(cropped_image, output_name);
disp(['The cropped image was saved as: ', output_name]);
