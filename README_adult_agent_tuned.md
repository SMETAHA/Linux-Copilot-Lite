# Adult Linux Agent Tuned

Локальный CLI/GUI dev-agent на LangGraph + Ollama.

Эта версия добавляет ручные настройки генерации модели поверх full-билда:

- `model` — любая модель Ollama, например `gemma4:e2b` или `gemma4:e4b`
- `num_predict` — лимит генерируемых токенов на каждый вызов модели
- `num_ctx` — размер контекстного окна Ollama
- `temperature`
- `top_p`
- `top_k`
- `repeat_penalty`
- `seed`
- `keep_alive`
- `num_gpu` — опциональная подсказка Ollama по GPU layers / GPU offload
- `num_thread` — опциональное число CPU-потоков
- `num_batch` — опциональный batch size Ollama
- `max_steps` — максимум циклов planner/tool за один запрос
- `critic pass` — финальная проверка ответа, можно выключить ради скорости

## Установка

```bash
pip install -r requirements_adult_agent_tuned.txt
```

На Arch/CachyOS для GUI может понадобиться:

```bash
sudo pacman -S tk
```

## CLI запуск

Быстрый вариант:

```bash
python adult_linux_agent_tuned.py --workspace ~/Projects/clipx --model gemma4:e2b
```

Более подробные ответы:

```bash
python adult_linux_agent_tuned.py \
  --workspace ~/Projects/clipx \
  --model gemma4:e4b \
  --num-predict 900 \
  --num-ctx 8192 \
  --temperature 0 \
  --top-p 0.9 \
  --top-k 40 \
  --repeat-penalty 1.08 \
  --num-gpu 999 \
  --max-steps 10 \
  --keep-alive 15m
```

Отключить critic-pass для скорости:

```bash
python adult_linux_agent_tuned.py --disable-critic
```

## GUI запуск

```bash
python adult_linux_agent_tuned_gui.py
```

В GUI сверху есть поля:

- модель
- workspace
- runtime-dir
- `num_predict`
- `num_ctx`
- `temperature`
- `top_p`
- `top_k`
- `repeat_penalty`
- `max_steps`
- `seed`
- `keep_alive`
- `num_gpu`
- `num_thread`
- `num_batch`
- `critic pass`

После изменения настроек нажми **Запустить / перезапустить**.

## Рекомендованные настройки

Для быстрых tool-calling задач:

```text
model: gemma4:e2b
num_predict: 512
num_ctx: 4096
temperature: 0
top_p: 0.9
top_k: 40
repeat_penalty: 1.1
max_steps: 8
critic: on
```

Для более длинного анализа проекта:

```text
model: gemma4:e4b
num_predict: 900
num_ctx: 8192
temperature: 0
top_p: 0.9
top_k: 40
repeat_penalty: 1.08
max_steps: 10
critic: on
```

Если ответы всё ещё короткие, первым делом увеличивай `num_predict`: 512 → 900 → 1400.
Если агент не успевает изучить проект, увеличивай `max_steps`: 8 → 10 → 12.
Если модель слишком творит отсебятину, держи `temperature=0`.

## Slash-команды CLI

```text
/status
/settings
/memory
/history
/summary
/index
/index-search QUERY
/git
/diff
/backups
/clear-history
/exit
```


## Про num_gpu

`num_gpu` можно оставить пустым — Ollama сама решит, сколько слоёв держать на GPU.
Если хочешь попробовать вытолкнуть больше слоёв на видеокарту, укажи большое значение, например `999`.
Если VRAM не хватает, Ollama всё равно может откатить часть нагрузки на CPU. Проверяй через:

```bash
ollama ps
```
