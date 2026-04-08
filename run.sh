python /work/hdd/bfga/yyu69/qos-qoe-translation/src/relationship_extraction.py \
  --md /projects/bfga/yyu69/temp/0.md \
  --prompt_file /work/hdd/bfga/yyu69/qos-qoe-translation/src/prompts/relationship_extraction.txt \
  --out /projects/bfga/yyu69/temp/relationship_extraction_0.json \
  --paper_csv /projects/bfga/yyu69/datas/papers/mmsys_papers_related_to_qos-qoe_translation.csv

python /work/hdd/bfga/yyu69/qos-qoe-translation/src/metadata_enrichment.py \
  --md /projects/bfga/yyu69/temp/0.md \
  --rels /projects/bfga/yyu69/temp/relationship_extraction_0.json \
  --metadata_prompt_file /work/hdd/bfga/yyu69/qos-qoe-translation/src/prompts/metadata_enrichment.txt \
  --out /projects/bfga/yyu69/temp/metadata_enrichment_0.json

python /work/hdd/bfga/yyu69/qos-qoe-translation/src/data_evaluation.py \
  --input /projects/bfga/yyu69/temp/metadata_enrichment_0.json \
  --output /projects/bfga/yyu69/temp/data_evaluation_0.json \
  --system-prompt /work/hdd/bfga/yyu69/qos-qoe-translation/src/prompts/data_evaluation.txt \
  --reviewers claude,gemini,grok \
  --claude-model claude-haiku-4-5-20251001 \
  --gemini-model gemini-2.5-flash-lite \
  --grok-model grok-4.20-0309-reasoning \
  --start 0 \
  --end 1

python /work/hdd/bfga/yyu69/qos-qoe-translation/src/data_evaluation.py \
  --input /projects/bfga/yyu69/temp/metadata_enrichment_0.json \
  --output /projects/bfga/yyu69/temp/data_evaluation_0.json \
  --system-prompt /work/hdd/bfga/yyu69/qos-qoe-translation/src/prompts/data_evaluation.txt \
  --reviewers gemini \
  --gemini-model gemini-2.5-flash-lite \
  --start 1 \
  --end 2

python /work/hdd/bfga/yyu69/qos-qoe-translation/src/main.py \
  --md /projects/bfga/yyu69/temp/2.md \
  --relationship_prompt /work/hdd/bfga/yyu69/qos-qoe-translation/src/prompts/relationship_extraction.txt \
  --paper_csv /projects/bfga/yyu69/datas/papers/mmsys_papers_related_to_qos-qoe_translation.csv \
  --metadata_prompt /work/hdd/bfga/yyu69/qos-qoe-translation/src/prompts/metadata_enrichment.txt \
  --evaluation_system_prompt /work/hdd/bfga/yyu69/qos-qoe-translation/src/prompts/data_evaluation.txt \
  --relationships_out /projects/bfga/yyu69/temp/relationship_extraction_2.json \
  --enriched_out /projects/bfga/yyu69/temp/metadata_enrichment_2.json \
  --evaluation_out /projects/bfga/yyu69/temp/data_evaluation_2.json \
  --reviewers claude,gemini,grok \
  --claude-model claude-haiku-4-5-20251001 \
  --gemini-model gemini-2.5-flash-lite \
  --grok-model grok-4.20-0309-reasoning