<script lang="ts">
  import { onMount } from "svelte";
  import ThemeButton from "./components/ThemeButton.svelte";
  import { HeartSolid } from 'flowbite-svelte-icons';
  import ThemeInputField from "./components/ThemeInputField.svelte";
  import Popup from "./components/Popup.svelte";
  import SideDrawer from "./components/SideDrawer.svelte";

  let rand: string = $state("-1");
  let popupOpen: boolean = $state(false);

  function getRand() {
    fetch("/api/rand")
      .then(d => d.text())
      .then(d => (rand = d));
  }

  onMount(getRand)

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

  {#if popupOpen}
      <Popup onClose={() => setPopupState(false)} >
        <div class="text-center min-h-[10rem] min-w-[20rem]">
          <h1 class="text-xl font-bold uppercase underline my-2">This is popup</h1>
          <span>These are children house by popup</span>
        </div>
      </Popup>
  {/if}

  <SideDrawer width={"25rem"} >
    <div class="flex flex-col justify-evenly items-center h-[50vh] border border-dashed border-theme p-2">
      <h1 class="font-bold uppercase underline text-xl my-2">Output</h1>
      <div class="p-4 rounded-md bg-theme text-white leading-7 tracking-widest">
        <span class="">This is Side Drawer Content</span>
        <span>It will contain the output probability stuffs and others...</span>
      </div>
    </div>
  </SideDrawer>
  
  <div>
    <ThemeInputField inputEvent={()=>getRand} />
    <ThemeButton label="Generate" />
  </div>
</section>