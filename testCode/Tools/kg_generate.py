import asyncio
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.App import App
from src.config import settings


async def build_kg_with_one_json(json_file_path: str):
    app = App(settings)

    try:
        await app.insert_one_json_file(json_file_path)
        print("Knowledge graph construction completed.")
        await app.close_cache_only()
        print("Cache saved.")
    except KeyboardInterrupt:
        print("\nInterrupted by user, saving cache...")
        await app.close_cache_only()
        print("Cache saved.")
    except Exception as e:
        print(f"Construction failed: {e}")
        await app.close_cache_only()
        print("Cache saved.")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build KG from one JSON file.")
    parser.add_argument(
        "--input-path",
        default=settings.root_dir + "/Input/dataset_for_hipporag2/sample_corpus.json",
        help="Path to input corpus JSON file.",
    )
    args = parser.parse_args()
    asyncio.run(build_kg_with_one_json(args.input_path))
