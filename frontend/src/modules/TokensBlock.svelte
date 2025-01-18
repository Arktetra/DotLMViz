<script lang="ts">
  import DottedBlockBase from "../components/DottedBlockBase.svelte";
  import Popup from "../components/Popup.svelte";

  let { tokens, tokenInd = $bindable(), children = null } = $props()

  let popUpEnable: boolean = $state(false)

  const MAX_TOKEN_SIZE = 5
  const MAX_TOKEN_COUNT = 10

  const setPopUpState = (state : boolean) => {
    popUpEnable = state;
  }

  const tokenClick = (i : number) => {
    popUpEnable = true
    tokenInd = i
  }
</script>


<DottedBlockBase label="Tokens" inStyle="min-w-[5rem] min-h-[5rem] flex-col items-center">
  {#each tokens as token, ind}
    <button 
      onclick={() => tokenClick(ind)}
      class="block text-xl text-theme font-bold my-1 hover:text-theme-alt hover:scale-[115%]"
    >
      {token.length > MAX_TOKEN_SIZE ? token.slice(0, MAX_TOKEN_SIZE) + ".." : token}
    </button>
  {/each}
  {#if popUpEnable}
    <Popup onCloseCb={() => setPopUpState(false)}>
      {#if children}
        {@render children()}
      {/if}
    </Popup>
  {/if}
</DottedBlockBase>