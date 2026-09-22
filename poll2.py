import urllib.request, json, time, subprocess

RENDER_KEY = "rnd_sQNQEKxZLYoG4nzlA9A4lbYp6Zt0"
SERVICE = "srv-dap49mmgekts73fo5220"
URL = "https://wechat-rewriter-backend.onrender.com/health"

def latest_status():
    req = urllib.request.Request(
        f"https://api.render.com/v1/services/{SERVICE}/deploys?limit=1",
        headers={"Authorization": "Bearer "+RENDER_KEY})
    d = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
    dep = d[0]["deploy"]
    return dep["id"], dep["status"]

deadline = time.time() + 540
while time.time() < deadline:
    try:
        dep_id, st = latest_status()
    except Exception as e:
        st = f"err:{e}"; dep_id="?"
    print(time.strftime("%H:%M:%S"), "deploy", dep_id[:12], "status:", st)
    if st in ("live", "deployed"):
        # 探活
        try:
            code = subprocess.run(["curl","-s","--max-time","15","-o","/dev/null","-w","%{http_code}",URL],
                                  capture_output=True, text=True).stdout.strip()
            print("health HTTP:", code)
            if code == "200":
                print("BACKEND_LIVE")
                break
        except Exception as e:
            print("probe err", e)
        time.sleep(8)
    elif st == "build_in_progress":
        time.sleep(20)
    elif st in ("build_failed", "deploy_failed", "failed"):
        print("DEPLOY_FAILED_GIVEUP")
        break
    else:
        time.sleep(10)
else:
    print("TIMEOUT")
