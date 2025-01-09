<script lang="ts">
  import { onMount } from "svelte";
  import ThemeButton from "./components/ThemeButton.svelte";
  import { HeartSolid } from 'flowbite-svelte-icons';

  let rand: string = $state("-1");

  function getRand() {
    fetch("/api/rand")
      .then(d => d.text())
      .then(d => (rand = d));
  }

  onMount(getRand)
</script>

<section class="min-w-full min-h-screen flex flex-col justify-evenly items-center">
  <h1 class="font-bold uppercase text-theme">Your number is {rand}!</h1>

  <ThemeButton label="Click" clickEvent={getRand} >
    <HeartSolid fill="red" class="inline-block" />
  </ThemeButton>
</section>