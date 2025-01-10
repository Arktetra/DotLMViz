<script lang="ts">
  import { onMount } from "svelte";
  import ThemeButton from "./components/ThemeButton.svelte";
  import { HeartSolid } from 'flowbite-svelte-icons';
  import ThemeInputField from "./components/ThemeInputField.svelte";
  import Popup from "./components/Popup.svelte";

  let rand: string = $state("-1");
  let inputField: string = $state("None");
  let popupOpen: boolean = $state(false);

  function getRand() {
    fetch("/api/rand")
      .then(d => d.text())
      .then(d => (rand = d));
  }

  onMount(getRand)

  function inputChange(val : any) {
    inputField = val;
  } 

  function setPopupState(state : boolean)
  {
    popupOpen = state;
  }
</script>

<section class="min-w-full min-h-screen flex flex-col justify-evenly items-center">
  <h1 class="font-bold uppercase text-theme">Your number is {rand}!</h1>

  <ThemeButton label="Click" clickEvent={() => setPopupState(!popupOpen)} >
    <HeartSolid fill="red" class="inline-block" />
  </ThemeButton>

  {inputField}

  {#if popupOpen}
      <Popup onClose={() => setPopupState(false)} >
        <div class="text-center min-h-[10rem] min-w-[20rem]">
          <h1 class="text-xl font-bold uppercase underline my-2">This is popup</h1>
          <span>These are children house by popup</span>
        </div>
      </Popup>
  {/if}
  
  <div>
    <ThemeInputField inputEvent={inputChange} />
    <ThemeButton label="Generate" />
  </div>

  <div>
    <ThemeInputField label="Password" type="password" inputEvent={inputChange} />
    <ThemeButton label="pass" />
  </div>
</section>