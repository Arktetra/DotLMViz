<script lang="ts">
  import { onMount } from "svelte";
  import ThemeButton from "./components/ThemeButton.svelte";
  import { HeartSolid } from 'flowbite-svelte-icons';
  import ThemeInputField from "./components/ThemeInputField.svelte";

  let rand: string = $state("-1");
  let inputField: string = $state("None");

  function getRand() {
    fetch("/api/rand")
      .then(d => d.text())
      .then(d => (rand = d));
  }

  onMount(getRand)

  function inputChange(val : any) {
    inputField = val;
  } 
</script>

<section class="min-w-full min-h-screen flex flex-col justify-evenly items-center">
  <h1 class="font-bold uppercase text-theme">Your number is {rand}!</h1>

  <ThemeButton label="Click" clickEvent={getRand} >
    <HeartSolid fill="red" class="inline-block" />
  </ThemeButton>

  {inputField}
  
  <div>
    <ThemeInputField inputEvent={inputChange} />
    <ThemeButton label="Generate" />
  </div>

  <div>
    <ThemeInputField label="Password" type="password" inputEvent={inputChange} />
    <ThemeButton label="pass" />
  </div>
</section>