import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'qapha-welcome',
  imports: [RouterLink],
  template: `
<main class="welcome">
  <section class="hero">
    <span class="pill">Learning by watching</span>
    <h1>Qapha predicts what happens next, <span class="accent">and checks itself honestly.</span></h1>
    <p class="lead">
      It watches a scene, makes a sealed guess before each new frame arrives,
      then measures how wrong it was and learns from it. One step toward machines
      that understand the physical world well enough to act in it.
    </p>
    <div class="hero-actions">
      <a routerLink="/lab" class="big-button">Load the demo</a>
      <a routerLink="/how-it-works" class="ghost-button">How it works</a>
    </div>
  </section>

  <section class="steps">
    <article class="step">
      <span class="step-num">1</span>
      <h3>Watch</h3>
      <p>Meta's SAM 3.1 finds and follows the object in every frame.</p>
    </article>
    <article class="step">
      <span class="step-num">2</span>
      <h3>Predict</h3>
      <p>Each guess, and how sure it is, is sealed before the answer is shown.</p>
    </article>
    <article class="step">
      <span class="step-num">3</span>
      <h3>Check and learn</h3>
      <p>Every miss is scored. Qapha learns about its scene, like where the floor is.</p>
    </article>
  </section>

  <p class="built-with">
    Built with <b>Meta SAM 3.1</b> · <b>Meta Muse Spark</b> · Python · Angular
  </p>
</main>
`,
})
export class WelcomeComponent {}
