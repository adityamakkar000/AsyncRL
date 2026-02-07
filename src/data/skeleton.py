class AnnealedRLDataset:
    def __init__(self, gcs_path, mask_rate=0.0):
        """
        Args:
            gcs_path: Path to data in GCS like "gs://bucket/data.jsonl"
            mask_rate: How much to anneal (0.0 = full prompts, 1.0 = minimal)
        """
        self.data = self.load_from_gcs(gcs_path)
        self.mask_rate = mask_rate
    
    def load_from_gcs(self, gcs_path):
        # Download and parse data
        # Returns list of {"problem": ..., "answer": ...}
        pass
    
    def get_batch(self, batch_size):
        """
        Returns a batch of (prompt, answer) pairs
        """
        # Sample batch_size items
        sampled = random.sample(self.data, batch_size)
        
        batch = []
        for item in sampled:
            # Apply annealing to create prompt
            prompt = self.create_prompt(item, self.mask_rate)
            answer = item["answer"]
            
            batch.append({
                "prompt": prompt,
                "answer": answer
            })
        
        return batch
    
    def create_prompt(self, item, mask_rate):
        """
        Create prompt with annealing applied
        """
        if random.random() > mask_rate:
            # Show example
            return f"Example: 5+3=8\n\n{item['problem']}"
        else:
            # No example
            return item['problem']