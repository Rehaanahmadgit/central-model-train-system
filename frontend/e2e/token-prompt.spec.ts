import { expect, test } from "@playwright/test";

import { authDetailFrom } from "../src/lib/api";

test("authDetailFrom: missing vs rejected token", () => {
  expect(authDetailFrom({ error: { code: "ACCESS_TOKEN_REQUIRED", message: "need" } }, false)).toEqual({
    rejected: false,
    message: "need",
  });
  expect(authDetailFrom({ error: { code: "ACCESS_TOKEN_INVALID", message: "bad" } }, true).rejected).toBe(true);
  expect(authDetailFrom(null, true).message).toContain("rejected");
  expect(authDetailFrom(null, false).message).toContain("requires");
});

// Fresh instance (mocked REST): prompt appears, a wrong token shows the 401 banner,
// the right token unlocks. The server rules themselves are covered by test_auth.py.
test("fresh instance asks for the token, rejects a wrong one, accepts the right one", async ({ page }) => {
  const GOOD = "good-token-123";
  await page.route("**/api/v1/**", async (route) => {
    const sent = route.request().headers()["x-access-token"];
    if (sent === GOOD) return route.fulfill({ json: { items: [], total: 0, security: { token_required: true, warning: null } } });
    const invalid = Boolean(sent);
    return route.fulfill({
      status: 401,
      json: {
        error: {
          code: invalid ? "ACCESS_TOKEN_INVALID" : "ACCESS_TOKEN_REQUIRED",
          message: invalid
            ? "That access token was rejected (401). Use the token printed in the docker logs on first boot."
            : "This instance requires the access token (found in /data/access-token).",
          details: {},
        },
      },
    });
  });

  await page.goto("/");
  await page.evaluate(() => window.localStorage.removeItem("trainhub-access-token"));
  await page.reload();
  const prompt = page.getByTestId("token-prompt");
  await expect(prompt).toBeVisible();
  await expect(page.getByTestId("token-rejected")).toHaveCount(0); // nothing sent yet

  await page.getByTestId("token-input").fill("wrong-token");
  await page.getByTestId("token-submit").click();
  await expect(page.getByTestId("token-rejected")).toContainText("rejected (401)");

  await page.getByTestId("token-input").fill(GOOD);
  await page.getByTestId("token-submit").click();
  await expect(page.getByTestId("token-prompt")).toHaveCount(0);
});

// Real server with auth ON: set TRAINHUB_E2E_AUTH_URL + TRAINHUB_E2E_TOKEN (read from
// <home>/access-token). Skipped otherwise.
test("real instance with auth on: prompt, wrong token, right token", async ({ page }) => {
  const url = process.env.TRAINHUB_E2E_AUTH_URL;
  const token = process.env.TRAINHUB_E2E_TOKEN;
  test.skip(!url || !token, "needs TRAINHUB_E2E_AUTH_URL and TRAINHUB_E2E_TOKEN");
  await page.goto(`${url}/`);
  await expect(page.getByTestId("token-prompt")).toBeVisible();
  await page.getByTestId("token-input").fill("definitely-wrong");
  await page.getByTestId("token-submit").click();
  await expect(page.getByTestId("token-rejected")).toContainText("rejected");
  await page.getByTestId("token-input").fill(token as string);
  await page.getByTestId("token-submit").click();
  await expect(page.getByTestId("token-prompt")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "New training" })).toBeVisible();
});
