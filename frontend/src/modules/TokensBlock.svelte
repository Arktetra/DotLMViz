<script lang="ts">
  import DottedBlockBase from "../components/DottedBlockBase.svelte";
  import Popup from "../components/Popup.svelte";

  let popUpEnable: boolean = $state(false);

  const tokens = [
    "A",
    "B",
    "C",
    "D"
  ]

  let activeToken: string = $state(tokens[0]);

  function setPopUpState(t : string, state : boolean)
  {
    activeToken = t;
    popUpEnable = state;
  }
</script>


<div class="">
  <DottedBlockBase label="Tokens" style="min-w-[4rem] flex flex-col items-center justify-center">
    {#each tokens as token}
      <button 
        on:click|stopPropagation={() => setPopUpState(token, true)}
        class="block text-xl text-theme font-bold my-4 hover:scale-125"
      >
        {token}
      </button>
    {/each}
    {#if popUpEnable}
      <Popup onClose={() => popUpEnable = false}>
        <div class="w-[20rem] h-[20rem] flex flex-col justify-evenly items-center">
          <span class="block font-bold text-2xl">{activeToken}</span>
          <p>detailed info here.....</p>
          <p>detailed info here [][]</p>
        </div>
      </Popup>
    {/if}
  </DottedBlockBase>
</div>