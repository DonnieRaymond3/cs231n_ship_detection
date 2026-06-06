import argparse
import json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cat-id", type=int, default=1)
    args = ap.parse_args()

    d = json.load(open(args.inp))
    name = d["categories"][0].get("name", "ship") if d.get("categories") else "ship"
    d["categories"] = [{"id": args.cat_id, "name": name, "supercategory": "none"}]
    for a in d["annotations"]:
        a["category_id"] = args.cat_id
    json.dump(d, open(args.out, "w"))
    print(f"wrote {args.out}: {len(d['images'])} images, "
          f"{len(d['annotations'])} anns, category_id={args.cat_id}")

if __name__ == "__main__":
    main()
