import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    <header class="shell">
      <a routerLink="/" class="brand-link">
        <span class="mark">Q</span>
        <span class="brand-name">Qapha</span>
      </a>
      <nav class="nav-links">
        <a routerLink="/" routerLinkActive="active" [routerLinkActiveOptions]="{ exact: true }">Home</a>
        <a routerLink="/how-it-works" routerLinkActive="active">How it works</a>
        <a routerLink="/lab" class="nav-cta">Open the lab</a>
      </nav>
    </header>
    <router-outlet />
  `,
})
export class AppComponent {}
