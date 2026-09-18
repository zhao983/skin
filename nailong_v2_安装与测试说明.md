# 奶龙 CustomKnight v2：安装与测试说明

## 当前状态

- 文件级重建已完成，成品候选目录：`D:\skin\nailong_v2`
- 基于 `D:\skin\Default` 全量复制，共 207 个运行文件。
- 只修改了 9 个确认含玩家本体的 atlas；另外 198 个文件与 Default 的 SHA-256 完全一致。
- 本机没有找到 `hollow_knight.exe`、`CustomKnight.dll`、CustomKnight 皮肤目录或游戏日志，因此尚未完成实机运行与录屏验收。
- 在完成下述实机测试以前，本版本应视为“待实机验收候选”，不能标记为最终完成。

## 已重建的玩家 atlas

| 文件 | 已替换的玩家组件 | 用途 |
|---|---:|---|
| `Knight.png` | 518 | 待机、移动、跳跃/下落、空中动作、左右平 A、上劈、下劈、受击等核心动画 |
| `Sprint.png` | 28 | 独立 Sprint 材质动画 |
| `Cloak.png` | 25 | 披风/冲刺相关特殊表现 |
| `Shriek.png` | 2 | 尖啸法术中的玩家本体 |
| `Wings.png` | 9 | 翅膀/二段跳相关玩家本体 |
| `Webbed.png` | 27 | 束缚表现中的玩家本体 |
| `DreamArrival.png` | 12 | 梦境到达表现中的玩家本体 |
| `Dreamnail.png` | 18 | 梦钉表现中的玩家本体 |
| `Birthplace.png` | 74 | 出生地相关演出中的玩家本体 |

构建方法不是按图片尺寸或连通区域相似度匹配动作。每个替换目标必须同时满足 Default 原帧中的浅色面具、深色面部细节、眼孔语义和同一透明玩家组件；姿势、坐标、pivot 所依赖的 atlas 位置、骨钉和外部特效均沿用 Default。另有 4 张与巨大特效粘连的边界帧经过人工对照后局部补齐。

## 安装

1. 完全退出《空洞骑士》。
2. 打开当前 CustomKnight 的 `Skins` 目录。
3. 不要同时安装旧的 `nailong` 和新的 `nailong_v2`；把已安装目录里的旧测试版移出 `Skins`。
4. 将整个 `D:\skin\nailong_v2` 复制为 `Skins` 的直接子目录。
5. 重新启动游戏并明确选择 `nailong_v2`。最稳妥的方式是重启游戏；仅覆盖同名 PNG 后不重新加载，可能继续使用内存中的旧纹理。

## 必须录制的实机测试

建议连续录制 60 秒以上，并确保画面中始终能看清角色：

1. 原地待机至少 30 秒：不能出现骨钉向上的攻击姿势，也不能闪回白色小骑士。
2. 向左、向右移动各 5 秒。
3. 跳跃、上升、最高点和下落各覆盖数次。
4. 空中左右平 A 各至少 3 次。
5. 地面左右平 A 各至少 3 次。
6. 上劈至少 3 次：骨钉必须向上。
7. 下劈至少 3 次：骨钉必须向下。
8. Sprint/冲刺、二段跳、受击各至少 3 次。

通过标准：整段录像中正常形态白色小骑士出现次数为 0；待机姿势稳定且无骨钉；左右与上下攻击方向全部正确；角色没有被裁切、漂移或突然改变大小。

## 文件级 QA 记录

- `D:\skin\_v2_diagnostics\v2_build_report.json`
- `D:\skin\_v2_diagnostics\v2_transformed_components.csv`
- `D:\skin\_v2_diagnostics\v2_overview_Knight.jpg`
- `D:\skin\_v2_diagnostics\v2_overview_Sprint.jpg`
- `D:\skin\_v2_diagnostics\v2_residual_candidates.csv`

残留对照结果：Skadi 对照的 377 个玩家候选矩形中，正常白色小骑士残留为 0；未改的 29 个候选均为黑色 Shade/暗影形态。
