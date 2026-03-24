import { initBotId } from "botid/client/core";

initBotId({
  protect: [
    { path: "/api/transcribe/session", method: "POST" },
    { path: "/api/blob/upload", method: "POST" },
    { path: "/api/transcribe", method: "POST" },
    { path: "/api/blob/delete", method: "POST" },
  ],
});
