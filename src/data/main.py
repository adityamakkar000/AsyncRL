import grain

dataset = (
    grain.MapDataset.source([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    .shuffle(seed=42)  # Shuffles elements globally.
    .map(lambda x: x+1)  # Maps each element.
    .batch(batch_size=2)  # Batches consecutive elements.
)

for batch in dataset:
  print(batch)