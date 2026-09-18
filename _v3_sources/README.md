# v3 奶龙动作源说明

## 主要参考

- `D:\skin\奶龙\奶龙.png`
- `D:\skin\奶龙\奶龙02.png`
- `D:\skin\奶龙\奶龙03.png`
- `D:\skin\奶龙\部分动作参考图.png`
- `D:\skin\展示视频.mp4` 的放大抽帧

## 生成母图

- `nailong_movement_master.png`：12 个待机、走跑、跳跃、冲刺、受伤和倒地姿势。
- `nailong_attack_master.png`：6 个平 A、上劈、下劈和空中持骨钉姿势。
- `donors\`：从母图分割并恢复为真实透明 alpha 的 18 个动作源。
- `donor_contact.png`：动作源总览。

## 最终图像生成提示词

移动母图：

> Use case: production sprite-source sheet for a 2D CustomKnight game skin. Create one clean 4×3 sprite master sheet of the yellow Nailoong character, matching the supplied Nailoong references and the rounded full-body character visible in the gameplay crop. The character itself is the priority; do NOT preserve or imitate Hollow Knight's Knight silhouette. Character lock: one continuous plump pear-shaped yellow body with a very large rounded head flowing directly into the torso; huge vivid green eyes with black pupils and tiny white highlights; tiny friendly mouth; large cream oval belly; short thick rounded arms with dark gray fingertips; short thick legs and rounded feet; warm golden-yellow, soft dimensional 2D/3D hybrid shading, clean dark brown outline; no horns, no mask, no cloak, no cape, no orange tassels, no tiny human body, no stick limbs. Exactly 12 isolated full-body poses: front idle, right-facing side idle, two walk poses, run, jump rising, airborne neutral, falling, landing crouch, horizontal dash, hurt recoil, prone recovery. True transparent background, no scene, labels, text, grid, borders or overlapping poses.

攻击母图：

> Use case: production attack-sprite source sheet for a 2D CustomKnight game skin. Create one clean 3×2 sheet containing exactly six isolated full-body Nailoong attack poses. Match the Nailoong references and movement sheet exactly: huge green eyes, tiny mouth, one continuous plump pear-shaped golden-yellow body, large cream oval belly, short thick rounded arms/legs, dark gray fingertips, soft dimensional 2D/3D shading, clean dark brown outline. Use one simple Hollow-Knight-like silver nail weapon. Include right-facing wind-up, horizontal thrust, recovery, upward slash, downward midair slash and neutral airborne attack. No horns, mask, cloak, cape, orange tassels or Knight silhouette. Nothing cropped; true transparent background; no text, grid, scene or overlap.

图像生成器返回的预览包含烘焙棋盘格，`D:\skin\tools\prepare_nailong_v3_sources.py` 通过颜色锚点、连通体和孔洞填充将其恢复为真实透明 PNG；最终 atlas 使用的是 `donors\` 中的透明动作源。
