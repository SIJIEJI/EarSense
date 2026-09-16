import math
import pygame
import numpy as np
import pandas as pd
import random
import time
from scipy.signal import windows
from pygame.locals import *
class PVTTask:
    def __init__(self,
                 total_duration=300,
                 isi_range=(2.0, 10.0),
                 response_window=3.0,
                 num_blocks=1):
        pygame.init()
        pygame.key.set_repeat(0)  # avoid repeated keypress from key being held
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 1500, 900
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("PVT Task")

        # Colors
        self.WHITE  = (255, 255, 255)
        self.BLACK  = (0, 0, 0)
        self.RED    = (255, 0, 0)
        self.ORANGE = (255, 140, 0)

        # Timing & blocks
        self.total_duration = total_duration
        self.isi_min, self.isi_max = isi_range
        self.response_window = response_window
        self.num_blocks = num_blocks
        self.block_duration = total_duration / num_blocks
        self.feedback_ms = 250  # how long to flash the orange box after response

        # UI sizes
        self.BOX_W, self.BOX_H = 320, 160
        self.BOX_BORDER = 6

        # Fonts
        self.instr_font = pygame.font.Font(None, 36)
        self.timer_font = pygame.font.Font(None, 64)   # central timer
        self.small_font = pygame.font.Font(None, 36)   # last RT overlay

        # State
        self.last_rt_text = None
        self.quit_flag = False

        # Data storage
        self.trial_data = []

    # ---------- overlays ----------
    def _last_rt(self):
        """Top-right 'Last RT' overlay shown during ISI."""
        if not self.last_rt_text:
            return None
        surf = self.small_font.render(self.last_rt_text, True, self.BLACK)
        rect = surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(surf, rect)
        return rect

    # ---------- drawing ----------
    def _draw_fixation(self, show_last=True):
        cx, cy = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        self.screen.fill(self.WHITE)
        pygame.draw.line(self.screen, self.BLACK, (cx - 20, cy), (cx + 20, cy), 2)
        pygame.draw.line(self.screen, self.BLACK, (cx, cy - 20), (cx, cy + 20), 2)
        if show_last:
            self._last_rt()
        pygame.display.flip()

    def _draw_timer_box(self, ms_text: str, box_color):
        """
        Draws a centered timer string inside a colored-outlined box.
        """
        cx, cy = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        self.screen.fill(self.WHITE)

        # box rect centered
        rect = pygame.Rect(0, 0, self.BOX_W, self.BOX_H)
        rect.center = (cx, cy)

        # outline box
        pygame.draw.rect(self.screen, box_color, rect, self.BOX_BORDER)

        # timer text
        surf = self.timer_font.render(ms_text, True, self.BLACK)
        self.screen.blit(surf, surf.get_rect(center=(cx, cy)))

        pygame.display.flip()

    # ---------- event handling helpers ----------
    def _drain_isi_until(self, end_ts):
        """During ISI: show fixation + last RT, ignore all keypresses until end_ts."""
        self._draw_fixation(show_last=True)
        while not self.quit_flag and time.time() < end_ts:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.quit_flag = True
                # ignore all keys during ISI
            pygame.time.wait(10)

    def _clear_stale_keydowns(self):
        """Clear any key press that occurred before the response window."""
        pygame.event.clear([pygame.KEYDOWN, pygame.KEYUP])

    # ---------- UI ----------
    def show_instructions(self):
        lines = [
            "PVT",
            "As soon as the timer appears, press SPACE as quickly as possible.",
            "Your reaction time (ms) is shown in the box.",
            "Press SPACE to begin."
        ]
        self.screen.fill(self.WHITE)
        y = self.SCREEN_HEIGHT//2 - 60
        for line in lines:
            surf = self.instr_font.render(line, True, self.BLACK)
            self.screen.blit(surf, surf.get_rect(center=(self.SCREEN_WIDTH//2, y)))
            y += 40
        pygame.display.flip()
        while not self.quit_flag:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.quit_flag = True
                if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                    return
            pygame.time.wait(10)

    # ---------- main ----------
    def run(self):
        """Run the full session, returning a DataFrame of trial-by-trial data."""
        sync_start = time.time()
        self.show_instructions()
        trial = 0

        while True:
            now = time.time()
            elapsed = now - sync_start
            if elapsed >= self.total_duration or self.quit_flag:
                break

            trial += 1
            onset = now - sync_start
            block = min(int(onset // self.block_duration) + 1, self.num_blocks)

            # pre-stim short fixation that drains keys
            self._drain_isi_until(time.time() + 0.5)

            # start of the response window
            self._clear_stale_keydowns()

            stim_onset = time.time()
            responded, rt = False, None
            deadline = stim_onset + self.response_window

            # live timer loop (RED box)
            while time.time() < deadline and not self.quit_flag and not responded:
                elapsed_ms = (time.time() - stim_onset) * 1000.0
                ms_text = f"{elapsed_ms:,.0f} ms"
                self._draw_timer_box(ms_text, self.RED)

                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        self.quit_flag = True
                    elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                        responded = True
                        rt = time.time() - stim_onset
                pygame.time.wait(10)

            # prepare "last RT" label ONCE
            if responded and rt is not None:
                final_ms = int(rt * 1000)
                self.last_rt_text = f"Last RT: {final_ms} ms"

                # brief feedback flash with ORANGE box showing the frozen RT
                self._draw_timer_box(f"{final_ms:,d} ms", self.ORANGE)
                pygame.time.wait(self.feedback_ms)
            else:
                self.last_rt_text = "Last RT: — (miss)"

            # after feedback / timeout, go straight to fixation (shows last RT)
            self._draw_fixation(show_last=True)

            # log trial
            self.trial_data.append({
                "Block":          block,
                "Trial":          trial,
                "Onset_time(s)":  onset,
                "Stimulus_Shape": "timer_box",
                "Stimulus_Color": "red_outline",
                "Should_Press":   True,
                "Responded":      responded,
                "RT(s)":          rt,
                "Correct":        responded
            })

            # ISI (keys ignored)
            isi_end = time.time() + random.uniform(self.isi_min, self.isi_max)
            self._drain_isi_until(isi_end)

            if time.time() - sync_start >= self.total_duration:
                break

        # end-of-experiment screen
        self.screen.fill(self.WHITE)
        end_surf = self.instr_font.render("Experiment complete", True, self.BLACK)
        self.screen.blit(end_surf, end_surf.get_rect(center=(self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2)))
        pygame.display.flip()
        pygame.time.wait(2000)
        pygame.quit()

        return {"Trial Data": pd.DataFrame(self.trial_data)}