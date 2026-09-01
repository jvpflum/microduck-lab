# MicroDuck speed research scoreboard

| Policy | World-X m/s | Body m/s | Survival | Lateral m | Heading ° |
|---|---:|---:|---:|---:|---:|
| speed-scout-transfer-i160-linehold-frictionless | 1.9259187331419316 | 1.928319832485143 | 1.0 | 0.23765085978233264 | 3.14010099356734 |
| speed-scout-controller-baseline | 1.8635606903518558 | 1.8664952516563154 | 1.0 | 0.5630704540317064 | 3.3718299631015043 |
| speed-retention-v3-final-official | 1.0991368399104162 | 1.0997094490441068 | 1.0 | 0.15981195377299526 | 3.7626187427709503 |
| speed-scout-transfer-current-friction-0025 | 0.9972458522256129 | 1.0017459020789012 | 0.8 | 0.679271794782324 | 7.034508661502244 |
| official-sweep-balanced-lr5e-6-std0.10-s605-e4096-i350 | 0.9656577835429375 | 0.9691945660342425 | 1.0 | 0.5924557947380957 | 3.713120663522215 |
| official-sweep-balanced-lr2e-6-std0.06-s601-e4096-i350 | 0.842264968770682 | 0.847152436666061 | 1.0 | 0.6213494870451586 | 4.287234166888298 |
| official-sweep-line_hold-lr2e-6-std0.06-s604-e4096-i350 | 0.8421419106690954 | 0.8465294856359726 | 1.0 | 0.4961340270020656 | 3.6222375818505816 |
| official-sweep-speed_retention-lr1e-6-std0.06-s602-e4096-i350 | 0.8307980763189806 | 0.8360789987551921 | 1.0 | 0.5816767653500051 | 4.053127932752148 |
| official-sweep-speed_retention-lr2e-6-std0.03-s603-e4096-i350 | 0.7938369755301408 | 0.7988024909541221 | 1.0 | 0.4447589272455666 | 3.5597741702139074 |

## Historical transfer audit

{
  "available": true,
  "highest_stage": 5,
  "interpretation": "Prior transfer plateaued before official 0.003 friction; use low-plasticity gating rather than another direct full-drag fine-tune.",
  "last_friction": 0.0025,
  "last_stage": 5,
  "last_training_world_x_mps": 0.7623
}
