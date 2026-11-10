import { expect, test } from "@playwright/test";

import { boxRect } from "../src/lib/predictions";
import {
  annotatedFilename,
  checkFiles,
  visiblePredictions,
  type InferImage,
} from "../src/lib/tryit";

function file(name: string, type: string, size = 10): File {
  return new File([new Uint8Array(size)], name, { type });
}

test("only png/jpg/webp, 10 MB and 8 images are accepted, with a reason for each refusal", () => {
  const many = Array.from({ length: 10 }, (_, i) => file(`${i}.png`, "image/png"));
  const { ok, problems } = checkFiles([
    file("notes.txt", "text/plain"),
    file("huge.jpg", "image/jpeg", 11 * 1024 * 1024),
    ...many,
  ]);
  expect(ok).toHaveLength(8);
  expect(problems[0]).toContain("notes.txt: only PNG, JPG or WEBP");
  expect(problems[1]).toContain("huge.jpg");
  expect(problems[1]).toContain("10 MB");
  expect(problems.filter((p) => p.includes("at most 8"))).toHaveLength(2);
});

// Last-four rule: GT rows have 5 values, prediction rows 6.
test("confidence and class filters apply to 6-value rows; the box is the last four values", () => {
  const image: InferImage = {
    name: "a.png",
    width: 200,
    height: 100,
    pred: [
      [0, 0.9, 0.5, 0.5, 0.5, 0.5],
      [1, 0.4, 0.25, 0.25, 0.2, 0.2],
      [0, 0.1, 0.8, 0.8, 0.1, 0.1],
    ],
    gt: [[0, 0.5, 0.5, 0.5, 0.5]],
  };
  const names = ["cat", "dog"];
  expect(visiblePredictions(image, 0.25, "all", names)).toHaveLength(2);
  expect(visiblePredictions(image, 0.5, "all", names)).toHaveLength(1);
  expect(visiblePredictions(image, 0.0, "dog", names)).toEqual([image.pred[1]]);
  const rect = boxRect(image.pred[0], 200, 100);
  expect(rect).toEqual({ x: 50, y: 25, w: 100, h: 50 });
  // GT row (5 values) gives the same geometry.
  expect(boxRect(image.gt![0], 200, 100)).toEqual(rect);
});

test("annotated filename is safe and ends in _detected.png", () => {
  expect(annotatedFilename("my photo (1).jpg")).toBe("my_photo_1__detected.png");
  expect(annotatedFilename("../../x.png")).not.toContain("/");
  expect(annotatedFilename("../../x.png")).toMatch(/_detected\.png$/);
});
