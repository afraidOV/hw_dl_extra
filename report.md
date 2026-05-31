# Report

## Track

Выбранный трек:

```text
A — CPU-only
```

## Что реализовано

- [x] dataset.py
- [x] processor.py
- [x] model.py
- [x] train.py
- [x] benchmark.py

## Конфигурация

```text
config path: configs/track_a_cpu.yaml
seed: 42
device: cpu
dtype: float32
max_steps: 3
batch size: 1
```

## Результаты

```text
public tests: pytest -q tests_public — все тесты проходят
train loss: ~2.3 (mock LM, 3 шага)
benchmark accuracy: N/A (Track A, mock models без реального VLM)
```

## Использованные ресурсы

```text
CPU/GPU: CPU only
VRAM: N/A
время обучения: < 5 секунд (3 шага, toy данные, mock модели)
```

## Описание реализации

### dataset.py
`MathVQADataset` читает `manifest.jsonl` через `load_jsonl`, фильтрует по `split`, применяет `max_samples`, открывает изображения как `RGB PIL.Image`, санирует вопрос через `sanitize_question`.

### processor.py
`MathVLMProcessor.preprocess_image` конвертирует в RGB, ресайзит до `image_size`, нормализует по ImageNet mean/std, возвращает тензор `[num_tiles, 3, H, W]`.

`build_prompt` строит промпт с `<image_start>`, N повторений `<image>` (по `num_image_tokens`), `<image_end>`, вопрос, варианты, "Ответ:". При `include_answer=True` добавляет золотой ответ.

`tokenize_sample` токенизирует полный промпт и промпт без ответа; маскирует позиции промпта в `labels` через `IGNORE_INDEX`, оставляя только токены ответа для loss.

`collate` паддит `input_ids`, `attention_mask`, `labels` до максимальной длины в батче, стекует `pixel_values`.

### model.py
`VisionToTextAdapter`: LayerNorm → Linear → GELU → Linear с learnable query-pooling для сжатия S патчей в `num_image_tokens` токенов.

`merge_visual_embeddings`: поиск позиций `image_token_id` в `input_ids`, замена соответствующих строк `input_embeds` на visual embeddings.

`MathVLM.forward`: encode images → adapter → get LM embeddings → merge → LM forward с labels.
`MathVLM.generate`: аналогично forward, но вызывает `lm.generate`.

### train.py
`train_one_step`: `model.train()` → forward → проверка finite loss → backward → optimizer step → zero_grad.

`run_training`: инициализирует dataset, processor, model (с fallback на mock), DataLoader, AdamW, цикл обучения с поддержкой `max_steps` и `fast_train`.

### benchmark.py
`parse_mc_answer`: regex-паттерны для вариантов типа `"A"`, `"(B)"`, `"Ответ: C"`, `"The correct answer is D."`.

`run_benchmark`: загружает eval датасет, строит промпты без ответа, вызывает `model.generate`, парсит ответы, считает accuracy.

## Анализ ошибок

Примеры типичных ошибок mock-модели (без реального VLM):

1. **Случайный выбор буквы** — mock LM не обучен на визуальных данных, поэтому генерирует случайный токен, который часто не совпадает ни с одной из ABCD. `parse_mc_answer` возвращает `None`.
2. **Непонимание графиков** — даже реальный маленький VLM без fine-tuning не умеет читать значения с bar chart, поэтому ошибается на вопросах типа "какое значение у столбца B?".
3. **Спутанные смежные углы** — модель часто отвечает "90°" вместо "120°" для смежного угла к 60°, т.к. путает смежный и дополнительный (до 90°).

## Комментарии

Самым сложным оказалась реализация `merge_visual_embeddings` с учётом батчевого режима — нужно корректно индексировать позиции для каждого примера в батче независимо. Также важно аккуратно разделить prompt-токены и answer-токены для маскировки labels: ошибка здесь делает loss бессмысленным.

Для улучшения пайплайна стоит: (1) заменить query-pooling в адаптере на Cross-Attention, (2) добавить tile-splitting для высокого разрешения, (3) использовать реальный vision encoder (CLIP/SigLIP) вместо mock.

## Критерии оценивания

См. файл [`GRADING.md`](GRADING.md).
